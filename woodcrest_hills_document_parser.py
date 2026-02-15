from typing import Sequence, Any, List

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.node_parser import NodeParser, SemanticSplitterNodeParser
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import BaseNode
from llama_index.core.utils import get_tqdm_iterable
from pydantic import SerializeAsAny

from meeting_minutes_parser import MeetingMinutesParser


class WoodcrestHillsDocumentParser(NodeParser):
    llm: SerializeAsAny[LLM] = Field(
        description="LLM used to parse document"
    )
    embed_model: SerializeAsAny[BaseEmbedding] = Field(
        description="The embedding model to use to for semantic comparison",
    )

    def _parse_nodes(self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        meeting_node_parser = MeetingMinutesParser(llm=self.llm)
        semantic_splitter_node_parser = SemanticSplitterNodeParser(embed_model=self.embed_model)
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")
        for node in nodes_with_progress:
            all_nodes.extend(
                meeting_node_parser.get_nodes_from_documents(
                    [node], show_progress, **kwargs) if self.is_meeting_minutes_document(node)
                else semantic_splitter_node_parser.get_nodes_from_documents([node], show_progress, **kwargs)
            )
        return all_nodes

    def is_meeting_minutes_document(self, document: BaseNode) -> bool:
        response = self.llm.complete(
            prompt=f""""
            Answer with a simple yes or no. Is the following document a meeting minute document:
            {document.text}
            """)
        if "yes" in response.text.lower():
            return True
        return False


