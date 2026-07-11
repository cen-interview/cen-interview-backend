import anyio
import httpx
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from interview.config import settings


class EvidenceMcpClient:
    """Evidence 수집 단계에서 Notion/GitHub MCP tool call을 실행하는 얇은 래퍼.

    이 클래스는 MCP 연결과 tool 호출만 담당한다. 응답을 RawDoc으로 바꾸는 일은
    sources.py의 NotionSource/GitHubSource가 담당한다.
    """
    def __init__(
          self,
          notion_mcp_url: str | None = None,
          notion_access_token: str | None = None,
          timeout_seconds: float = 30.0,
      ) -> None:
          """MCP 호출에 필요한 endpoint와 인증 정보를 초기화한다.

          Args:
              notion_mcp_url: Notion MCP Streamable HTTP endpoint.
                  None이면 settings.notion_mcp_url을 사용한다.
              notion_access_token: Notion OAuth access token.
                  None이면 settings.notion_mcp_access_token을 사용한다.
                  실제 서비스에서는 사용자별 token을 주입하는 방식으로 전환한다.
              timeout_seconds: MCP 연결과 tool call에 사용할 timeout 초 단위 값.
          """
          self.notion_mcp_url = notion_mcp_url or settings.notion_mcp_url
          self.notion_access_token = notion_access_token or settings.notion_mcp_access_token
          self.timeout_seconds = timeout_seconds

    def list_notion_tools(self) -> list[dict]:
        """Notion MCP 서버가 제공하는 tool 목록과 입력 schema를 조회한다.

        실제 page/database 조회 tool 이름과 arguments 구조를 확정하기 전,
        개발자가 MCP 서버의 tool 계약을 확인할 수 있도록 사용한다.

        Returns:
            Notion MCP 서버가 노출하는 tool 목록. 각 항목에는 tool 이름,
            설명, input schema 등이 포함된다.

        Raises:
            ValueError:
                Notion MCP access token이 설정되어 있지 않은 경우.
        """
        if not self.notion_access_token:
            raise ValueError("Notion MCP 토큰이 필요합니다.")

        return anyio.run(self._list_notion_tools_async)
    
    def call_notion_tool(self, root_link: str, tool_name: str, arguments: dict | None = None) -> dict:
        """Notion MCP tool을 호출하고 원본 응답을 반환한다.

        Args:
            root_link: 사용자가 등록한 Notion page/database 링크.
            tool_name: list_notion_tools()로 확인한 실제 Notion 조회 tool 이름.
            arguments: tool 호출에 넘길 추가 arguments.
                None이면 root_link를 기본 argument로 전달한다.

        Returns:
            MCP tool call 원본 응답.
        """
        if not self.notion_access_token:
            raise ValueError("Notion MCP 토큰이 필요합니다.")

        tool_arguments = arguments or {"url": root_link}
        return anyio.run(self._call_notion_tool_async, tool_name, tool_arguments)

    @asynccontextmanager
    async def _notion_session(self) -> AsyncIterator[ClientSession]:
        """Notion MCP Streamable HTTP session을 열고 초기화한다.

        MCP 연결 생성, OAuth access token 주입, session.initialize() 호출을
        한곳에 모아 list_tools/call_tool 양쪽에서 같은 연결 절차를 재사용한다.
        """
        headers = {"Authorization": f"Bearer {self.notion_access_token}"}
        async with httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(self.timeout_seconds),
        ) as http_client:
            async with streamable_http_client(
                self.notion_mcp_url,
                http_client=http_client,
            ) as (read_stream, write_stream, _):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self.timeout_seconds),
                ) as session:
                    await session.initialize()
                    yield session

    async def _list_notion_tools_async(self) -> list[dict]:
        """초기화된 Notion MCP session으로 tool 목록을 비동기 조회한다."""
        async with self._notion_session() as session:
            tools = await session.list_tools()
            return [tool.model_dump(mode="json") for tool in tools.tools]


    async def _call_notion_tool_async(self, tool_name: str, arguments: dict) -> dict:
        """초기화된 Notion MCP session으로 지정 tool을 비동기 호출한다."""
        async with self._notion_session() as session:
            result = await session.call_tool(tool_name, arguments=arguments)
            return result.model_dump(mode="json")

    def call_github_tool(self, repo_link: str) -> dict:
        """GitHub MCP tool을 호출하고 원본 응답을 반환한다.

        TODO(담당 A):
            - MCP client/session 연결
            - repo contents/tree 조회 tool 호출
            - timeout/retry/error 처리
        """
        raise NotImplementedError
