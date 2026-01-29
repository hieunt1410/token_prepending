import torch
import torch.nn.functional as F
import numpy as np
import logging

from typing import TYPE_CHECKING, Dict, List
from tqdm import tqdm
from transformers import AutoTokenizer
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

from senllm import LlamaForCausalLM, Qwen2ForCausalLM, Gemma2ForCausalLM

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)


class TokenPrependingRetrievalModel:
    """Wraps a Token Prepending LLM for MTEB retrieval evaluation."""

    def __init__(self, args):
        self.args = args
        self.model_name_or_path = args.model_name_or_path

        # Load model
        model_path_lower = args.model_name_or_path.lower()
        if "llama" in model_path_lower:
            self.model = LlamaForCausalLM.from_pretrained(
                args.model_name_or_path,
                device_map="auto",
                output_hidden_states=True,
                trust_remote_code=True,
            )
        elif "qwen2" in model_path_lower:
            self.model = Qwen2ForCausalLM.from_pretrained(
                args.model_name_or_path,
                device_map="auto",
                output_hidden_states=True,
                trust_remote_code=True,
            )
        elif "gemma" in model_path_lower:
            self.model = Gemma2ForCausalLM.from_pretrained(
                args.model_name_or_path,
                device_map="auto",
                output_hidden_states=True,
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unsupported model: {args.model_name_or_path}")

        # Configure token prepending
        self.model.model.plan = args.use_which_plan
        self.model.model.tp_starting_index = args.tp_starting_index
        self.model.model.tp_exiting_index = args.tp_exiting_index

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

        # Add placeholder token for TP mode
        if args.use_which_plan == "tp":
            placeholder_token = "<PST>"
            self.tokenizer.add_tokens([placeholder_token])
            self.model.resize_token_embeddings(len(self.tokenizer))

            embedding_layer = self.model.get_input_embeddings()
            embedding_layer.weight.requires_grad_(False)
            placeholder_token_id = self.tokenizer.convert_tokens_to_ids(placeholder_token)
            num_dim = embedding_layer.weight.shape[1]
            device = embedding_layer.weight.device
            with torch.no_grad():
                embedding_layer.weight[placeholder_token_id] = torch.randn(
                    num_dim, device=device
                )
            embedding_layer.weight.requires_grad_(True)

        self.model.eval()

        # Config
        self.output_layer = args.output_layer
        self.encode_max_length = getattr(args, "encode_max_length", 512)
        self.prompt_method = args.prompt_method
        self.use_tp = args.use_which_plan == "tp"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _build_prompt(self, text: str) -> str:
        """Build the prompt-wrapped text for embedding extraction."""
        # Clean text
        if len(text) > 0 and text[-1] not in '.?"\'':
            text += "."
        text = text.replace('"', "'")
        if len(text) > 0 and text[-1] == "?":
            text = text[:-1] + "."

        if self.prompt_method == "prompteol":
            if self.use_tp:
                return f'Given a legal case, retrieve documents that are most similar to the case <PST> "{text}"'
            else:
                return f'Given a legal case, retrieve documents that are most similar to the case "{text}"'
        elif self.prompt_method == "cot":
            if self.use_tp:
                return f'After thinking step by step , this sentence : <PST> "{text}" means in one word:"'
            else:
                return f'After thinking step by step , this sentence : "{text}" means in one word:"'
        elif self.prompt_method == "ke":
            if self.use_tp:
                return f'The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this sentence : <PST> "{text}" means in one word:"'
            else:
                return f'The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this sentence : "{text}" means in one word:"'
        else:
            raise ValueError(f"Unknown prompt_method: {self.prompt_method}")

    @torch.no_grad()
    def _do_encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        all_embeds = []
        for start_idx in tqdm(
            range(0, len(texts), batch_size), desc="encoding", mininterval=10
        ):
            batch_texts = texts[start_idx : start_idx + batch_size]
            batch = self.tokenizer.batch_encode_plus(
                batch_texts,
                return_tensors="pt",
                padding=True,
                max_length=self.encode_max_length,
                truncation=True,
            )
            for k in batch:
                if batch[k] is not None:
                    batch[k] = batch[k].to(self.device)

            outputs = self.model(
                output_hidden_states=True, return_dict=True, **batch
            )
            hidden_states = outputs.hidden_states
            # Extract last token embedding from the specified layer
            embeds = hidden_states[self.output_layer][:, -1, :]
            embeds = F.normalize(embeds, p=2, dim=-1)

            if embeds.dtype == torch.bfloat16:
                embeds = embeds.float()

            all_embeds.append(embeds.cpu().numpy())
            del outputs, embeds, batch
            torch.cuda.empty_cache()

        return np.concatenate(all_embeds, axis=0)

    def encode_queries(
        self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> np.ndarray:
        prompted = [self._build_prompt(q) for q in queries]
        return self._do_encode(prompted, batch_size)

    def encode_corpus(
        self, corpus: List[Dict[str, str]], batch_size: int = 16, **kwargs
    ) -> np.ndarray:
        texts = [
            "{} {}".format(doc.get("title", ""), doc["text"]).strip()
            for doc in corpus
        ]
        prompted = [self._build_prompt(t) for t in texts]
        return self._do_encode(prompted, batch_size)

    def encode(
        self,
        inputs: "DataLoader",
        *,
        task_metadata: "TaskMetadata" = None,
        hf_split: str = None,
        hf_subset: str = None,
        prompt_type: PromptType = None,
        **kwargs,
    ) -> np.ndarray:
        """MTEB v2 encode interface."""
        batch_size = kwargs.get("batch_size", self.args.batch_size)

        all_texts = []
        for batch in inputs:
            if isinstance(batch, dict):
                if "text" in batch:
                    texts = batch["text"]
                elif "texts" in batch:
                    texts = batch["texts"]
                else:
                    for key, value in batch.items():
                        if (
                            isinstance(value, (list, tuple))
                            and len(value) > 0
                            and isinstance(value[0], str)
                        ):
                            texts = value
                            break
                    else:
                        raise ValueError(
                            f"Could not find text field in batch: {batch.keys()}"
                        )
            elif isinstance(batch, (list, tuple)):
                texts = batch
            else:
                texts = [batch]

            if isinstance(texts, str):
                texts = [texts]
            all_texts.extend(texts)

        if not all_texts:
            return np.array([])

        if prompt_type == PromptType.query:
            return self.encode_queries(all_texts, batch_size=batch_size)
        else:
            corpus = [{"text": text} for text in all_texts]
            return self.encode_corpus(corpus, batch_size=batch_size)

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return ModelMeta(
            name=self.model_name_or_path,
            revision=None,
            release_date=None,
            languages=None,
            loader=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=self.encode_max_length,
            embed_dim=self.model.config.hidden_size,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            similarity_fn_name="cosine",
            use_instructions=None,
            training_datasets=None,
        )

    def similarity(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> np.ndarray:
        e1 = np.asarray(embeddings1)
        e2 = np.asarray(embeddings2)
        e1 = e1 / np.linalg.norm(e1, axis=-1, keepdims=True)
        e2 = e2 / np.linalg.norm(e2, axis=-1, keepdims=True)
        return np.dot(e1, e2.T)

    def similarity_pairwise(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> np.ndarray:
        e1 = np.asarray(embeddings1)
        e2 = np.asarray(embeddings2)
        e1 = e1 / np.linalg.norm(e1, axis=-1, keepdims=True)
        e2 = e2 / np.linalg.norm(e2, axis=-1, keepdims=True)
        return np.sum(e1 * e2, axis=-1)
