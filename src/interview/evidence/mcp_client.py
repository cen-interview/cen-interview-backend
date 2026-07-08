class EvidenceMcpClient:
    """Evidence 수집 단계에서 Notion/GitHub MCP tool call을 실행하는 얇은 래퍼.

    이 클래스는 MCP 연결과 tool 호출만 담당한다. 응답을 RawDoc으로 바꾸는 일은
    sources.py의 NotionSource/GitHubSource가 담당한다.
    """

    def call_notion_tool(self, root_link: str) -> dict:
        """Notion MCP tool을 호출하고 원본 응답을 반환한다.

        TODO(담당 A):
            - MCP client/session 연결
            - Notion page/database 조회 tool 호출
            - timeout/retry/error 처리
        """
        raise NotImplementedError

    def call_github_tool(self, repo_link: str) -> dict:
        """GitHub MCP tool을 호출하고 원본 응답을 반환한다.

        TODO(담당 A):
            - MCP client/session 연결
            - repo contents/tree 조회 tool 호출
            - timeout/retry/error 처리
        """
        raise NotImplementedError
