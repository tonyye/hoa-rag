import json
from datetime import datetime, time, date
from typing import List, Any, Sequence

from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import LLM
from llama_index.core.llms.structured_llm import StructuredLLM
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.utils import get_tqdm_iterable
from pydantic import BaseModel, SerializeAsAny


class MeetingMinutes(BaseModel):
    title: str = Field(description="Title of the document")
    meeting_date: date = Field(description="The date this meeting was held")
    sub_title: str = Field(description="Subtitle of the document")
    location: str = Field(description="The location where the meeting was held")
    call_to_order_time: time = Field(description="Call to order time")
    board_members_present: list[str] = Field(description="List of board members who attended the meeting")
    board_members_absent: list[str] = Field(description="List of board members who did not attended the meeting")
    also_present: list[str] = Field(description="Additional people who attended the meeting")
    homeowner_forum: list[str] = Field(description="Items brought forth by homeowners in the meeting")
    business_items_architectural: list[str] = Field(description="Architectural business items discussed in the meeting")
    minutes: list[str] = Field(description="Minuted from the meeting")
    financials: list[str] = Field(description="Financials discussed in the meeting")
    delinquency_violations: list[str] = Field(description="Delinquency/Violations discussed in the meeting")
    other_business: list[str] = Field(description="Other business items discussed in the meeting")
    next_meeting_date: datetime = Field(description="Next meeting date")
    adjournment_time: time = Field(description="Time the meeting was adjourned")
    submitted_by: str = Field(description="Meeting minutes submitted by")

class MeetingMinutesParser(NodeParser):
    llm: SerializeAsAny[LLM] = Field(
        description="Structured LLM used to parse document"
    )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")
        sllm = self.llm.as_structured_llm(MeetingMinutes)

        for node in nodes_with_progress:
            nodes = self.get_nodes_from_node(node, sllm, **kwargs)
            all_nodes.extend(nodes)

        return all_nodes

    @staticmethod
    def get_nodes_from_node(document: BaseNode, sllm: StructuredLLM, **kwargs) -> List[BaseNode]:
        response = sllm.complete(document.text)
        meeting_minutes = response.raw
        meeting_date = f"{meeting_minutes.meeting_date: %A %B %d, %Y}"
        meeting_minutes_dict = json.loads(response.text)
        items = ["board_members_present", "also_present", "board_members_absent", "next_meeting_date", "submitted_by"]
        list_items = ["homeowner_forum", "business_items_architectural", "minutes",
                      "financials", "delinquency_violations", "other_business"]
        metadata = document.metadata | {
            "meeting_date": meeting_date
        }
        nodes = [
            TextNode(
                text=f"{meeting_minutes.title} was held on {meeting_date} from {meeting_minutes.call_to_order_time} to {meeting_minutes.adjournment_time}",
                metadata=metadata,
            ),
            TextNode(
                text=f"{meeting_minutes.title} was held at {meeting_minutes.location}",
                metadata=metadata,
            ),
        ]
        for item in items:
            nodes.append(
                TextNode(
                    text=f"""
                        {MeetingMinutes.model_fields[item].description} on {meeting_date}:
                        {meeting_minutes_dict[item]}
                    """,
                    metadata=metadata,
                )
            )
        for list_item in list_items:
            nodes.extend([
                TextNode(
                    text=f"""
                        {MeetingMinutes.model_fields[list_item].description} on {meeting_date}:
                        {item}
                    """,
                    metadata=metadata
                ) for item in meeting_minutes_dict[list_item]
            ])

        return nodes