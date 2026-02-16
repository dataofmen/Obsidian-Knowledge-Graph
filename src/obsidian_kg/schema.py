"""Ontology schema for Obsidian Knowledge Graph.

Defines entity types and relation types used during LLM-based extraction.
"""

from __future__ import annotations

from enum import Enum


class EntityType(str, Enum):
    """Types of entities extracted from Obsidian notes."""

    PERSON = "Person"
    PROJECT = "Project"
    CONCEPT = "Concept"
    TECHNOLOGY = "Technology"
    MEETING = "Meeting"
    DECISION = "Decision"
    RESOURCE = "Resource"


class RelationType(str, Enum):
    """Types of relations between entities."""

    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    CONTRADICTS = "CONTRADICTS"
    SUPPLEMENTS = "SUPPLEMENTS"
    DEPENDS_ON = "DEPENDS_ON"
    DECIDED_BY = "DECIDED_BY"
    CREATED_BY = "CREATED_BY"
    REFERENCES = "REFERENCES"


# Prompt fragment for LLM entity extraction
EXTRACTION_PROMPT = """다음 텍스트에서 주요 엔티티(개체)와 그들 간의 관계를 추출하세요.

## 엔티티 타입
- Person: 사람, 팀원, 이해관계자
- Project: 프로젝트, 작업, 업무
- Concept: 개념, 아이디어, 원칙
- Technology: 기술, 도구, 프레임워크, 라이브러리
- Meeting: 회의, 미팅, 워크숍
- Decision: 결정사항, 합의
- Resource: 문서, 링크, 참고자료

## 관계 타입
- RELATED_TO: 일반적 연관
- PART_OF: 하위 구성요소
- CONTRADICTS: 상충/모순
- SUPPLEMENTS: 보완
- DEPENDS_ON: 의존
- DECIDED_BY: 결정 주체
- CREATED_BY: 생성 주체
- REFERENCES: 참조

기존 지식 그래프의 노드와 어떤 관계(모순, 보완, 하위 개념)가 있는지 정의하세요.
추측하지 말고, 텍스트에 명시적으로 나타난 정보만 추출하세요.
"""
