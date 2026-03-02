"""MCP Tool registrations for OneDrive & SharePoint.
Import and call register_microsoft_tools(mcp, graph_client, auth_manager)
from your existing mcp_server.py

v1.9.2: Fixed auth status to check Postgres. Replaced fire-and-forget polling
         with explicit two-step flow (ms_auth_start + ms_auth_poll).
         Added try/except to all auth tools.
v1.9.3: Updated onedrive_download_file and sharepoint_download_file to save
         files directly to sandbox. Added session_id parameter.
v1.9.4: Made user_id optional on ALL Microsoft tools — auto-resolves from
         auth_manager.get_default_user_id_async() when not provided.
         Added resolve_share_link tool (22 tools total).
"""

import json
import logging

logger = logging.getLogger(__name__)


def register_microsoft_tools(mcp, graph_client, auth_manager):
    """Register all Microsoft OneDrive + SharePoint tools on the MCP server."""

    # -- Helper: resolve user_id --

    async def _resolve_user(user_id: str = None) -> str:
        """Resolve user_id: use provided value or fall back to last authenticated user.
        
        v1.9.4: This prevents 'Missing required parameter: user_id' errors
        when the MCP client doesn't pass user_id.
        """
        if user_id:
            return user_id
        
        default = await auth_manager.get_default_user_id_async()
        if default:
            logger.info(f"Auto-resolved user_id to: {default}")
            return default
        
        raise ValueError(
            "No user_id provided and no authenticated user found. "
            "Please authenticate first with ms_auth_start(user_id='your@email.com')"
        )

    # -- AUTH TOOLS --

    @mcp.tool()
    async def ms_auth_status(user_id: str = None) -> str:
        """Check Microsoft 365 authentication status for OneDrive/SharePoint access.

        Args:
            user_id: Microsoft email (e.g. user@company.com). Optional if already authenticated.
        """
        try:
            uid = await _resolve_user(user_id)
            authenticated = await auth_manager.is_authenticated_async(uid)
            return json.dumps({
                "authenticated": authenticated,
                "user_id": uid,
                "message": (
                    "Authenticated and ready for OneDrive/SharePoint."
                    if authenticated
                    else "Not authenticated. Use ms_auth_start to begin sign-in."
                ),
            }, indent=2)
        except ValueError as e:
            return json.dumps({
                "authenticated": False,
                "error": str(e),
                "message": "No authenticated user. Use ms_auth_start(user_id='your@email.com') first.",
            }, indent=2)
        except Exception as e:
            logger.error(f"ms_auth_status error: {e}", exc_info=True)
            return json.dumps({
                "authenticated": False,
                "error": str(e),
                "message": "Error checking auth status. Use ms_auth_start to re-authenticate.",
            }, indent=2)

    @mcp.tool()
    async def ms_auth_start(user_id: str) -> str:
        """Start Microsoft device login for OneDrive/SharePoint access.

        Returns a device code and URL. The user must visit the URL and enter
        the code. After entering the code, call ms_auth_poll to complete login.

        Args:
            user_id: Microsoft email (e.g. user@company.com) — REQUIRED for initial auth
        """
        try:
            result = await auth_manager.start_device_login(user_id)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"ms_auth_start error for {user_id}: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"Failed to start Microsoft login: {str(e)}",
            }, indent=2)

    @mcp.tool()
    async def ms_auth_poll(user_id: str = None) -> str:
        """Complete Microsoft device login after user has entered the code.

        Call this AFTER ms_auth_start, once the user confirms they have
        visited the URL and entered the device code.

        Args:
            user_id: Microsoft email. Optional if only one pending login.
        """
        try:
            uid = user_id
            if not uid:
                for u, data in auth_manager._tokens.items():
                    if data.get("status") == "pending":
                        uid = u
                        break
            if not uid:
                return json.dumps({
                    "status": "error",
                    "message": "No pending login found. Call ms_auth_start first.",
                }, indent=2)
            result = await auth_manager.poll_device_login(uid)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"ms_auth_poll error: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"Polling failed: {str(e)}",
            }, indent=2)

    # -- SHARING URL RESOLVER (v1.9.4) --

    @mcp.tool()
    async def resolve_share_link(
        sharing_url: str,
        save_to_sandbox: bool = True,
        user_id: str = None,
        session_id: str = "default",
    ) -> str:
        """Resolve a SharePoint or OneDrive sharing URL and optionally download the file.

        Takes a sharing link (the kind you copy from SharePoint/OneDrive "Share" button)
        and resolves it to the file's metadata including item_id, name, size, etc.

        When save_to_sandbox=True (default), also downloads the file directly to the
        sandbox so execute_code can access it immediately.

        WORKFLOW:
          1. User provides a sharing URL from SharePoint/OneDrive
          2. This tool resolves it to file metadata + downloads to sandbox
          3. Use execute_code to analyze: pd.read_excel('filename.xlsx')

        Args:
            sharing_url: Full sharing URL (e.g. https://company-my.sharepoint.com/:x:/p/user/ABC123)
            save_to_sandbox: Download file to sandbox (default True)
            user_id: Microsoft email. Optional if already authenticated.
            session_id: Sandbox session ID (default 'default')
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.resolve_share_link(
                uid, sharing_url,
                save_to_sandbox=save_to_sandbox,
                session_id=session_id,
            )
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({
                "error": True,
                "message": str(e),
            }, indent=2)
        except Exception as e:
            logger.error(f"resolve_share_link error: {e}", exc_info=True)
            return json.dumps({
                "error": True,
                "message": f"Failed to resolve sharing link: {str(e)}",
                "sharing_url": sharing_url,
            }, indent=2)

    # -- ONEDRIVE TOOLS --

    @mcp.tool()
    async def onedrive_list_files(
        user_id: str = None,
        path: str = "/",
        top: int = 50,
    ) -> str:
        """List files and folders in OneDrive.

        Args:
            user_id: Microsoft email. Optional if already authenticated.
            path: Folder path (/ for root, or /Documents/subfolder)
            top: Max items to return (default 50)
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.onedrive_list(uid, path, top)
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def onedrive_search(
        query: str,
        user_id: str = None,
        top: int = 25,
    ) -> str:
        """Search for files in OneDrive by name or content.

        Args:
            query: Search query string
            user_id: Microsoft email. Optional if already authenticated.
            top: Max results (default 25)
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.onedrive_search(uid, query, top)
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def onedrive_download_file(
        item_id: str,
        user_id: str = None,
        session_id: str = "default",
    ) -> str:
        """Download a file from OneDrive and save it to the sandbox.

        The file is written directly to the sandbox filesystem so that
        execute_code can immediately access it (e.g. pd.read_excel('filename.xlsx')).

        Args:
            item_id: The file's OneDrive item ID (from list, search, or resolve_share_link)
            user_id: Microsoft email. Optional if already authenticated.
            session_id: Sandbox session ID (default 'default')
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.onedrive_download(
                uid, item_id,
                save_to_sandbox=True,
                session_id=session_id,
            )
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def onedrive_upload_file(
        path: str,
        content_base64: str,
        user_id: str = None,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload a file to OneDrive (max 4MB for simple upload).

        Args:
            path: Destination path (e.g. /Documents/report.xlsx)
            content_base64: File content as base64-encoded string
            user_id: Microsoft email. Optional if already authenticated.
            content_type: MIME type (default application/octet-stream)
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.onedrive_upload(uid, path, content_base64, content_type)
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def onedrive_create_folder(
        folder_name: str,
        parent_path: str = "/",
        user_id: str = None,
    ) -> str:
        """Create a new folder in OneDrive.

        Args:
            folder_name: Name of the new folder
            parent_path: Parent folder path (/ for root)
            user_id: Microsoft email. Optional if already authenticated.
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.onedrive_create_folder(uid, parent_path, folder_name)
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def onedrive_delete_item(
        item_id: str,
        user_id: str = None,
    ) -> str:
        """Delete a file or folder from OneDrive.

        Args:
            item_id: The item's OneDrive ID
            user_id: Microsoft email. Optional if already authenticated.
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.onedrive_delete(uid, item_id)
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def onedrive_move_item(
        item_id: str,
        dest_folder_id: str,
        new_name: str = None,
        user_id: str = None,
    ) -> str:
        """Move a file or folder to a different OneDrive location.

        Args:
            item_id: Item to move
            dest_folder_id: Destination folder ID
            new_name: Optional new name for the item
            user_id: Microsoft email. Optional if already authenticated.
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.onedrive_move(uid, item_id, dest_folder_id, new_name)
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def onedrive_copy_item(
        item_id: str,
        dest_folder_id: str,
        new_name: str = None,
        user_id: str = None,
    ) -> str:
        """Copy a file or folder to a different OneDrive location.

        Args:
            item_id: Item to copy
            dest_folder_id: Destination folder ID
            new_name: Optional new name for the copy
            user_id: Microsoft email. Optional if already authenticated.
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.onedrive_copy(uid, item_id, dest_folder_id, new_name)
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def onedrive_share_item(
        item_id: str,
        share_type: str = "view",
        scope: str = "organization",
        user_id: str = None,
    ) -> str:
        """Create a sharing link for a OneDrive file or folder.

        Args:
            item_id: Item to share
            share_type: 'view' or 'edit' (default 'view')
            scope: 'anonymous', 'organization', or 'users' (default 'organization')
            user_id: Microsoft email. Optional if already authenticated.
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.onedrive_share(uid, item_id, share_type, scope)
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    # -- SHAREPOINT TOOLS --

    @mcp.tool()
    async def sharepoint_list_sites(
        search: str = None,
        user_id: str = None,
    ) -> str:
        """List or search SharePoint sites accessible to the user.

        Args:
            search: Optional search query to filter sites
            user_id: Microsoft email. Optional if already authenticated.
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.sharepoint_list_sites(uid, search)
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def sharepoint_get_site(
        site_id: str,
        user_id: str = None,
    ) -> str:
        """Get details of a specific SharePoint site.

        Args:
            site_id: SharePoint site ID
            user_id: Microsoft email. Optional if already authenticated.
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.sharepoint_get_site(uid, site_id)
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def sharepoint_list_drives(
        site_id: str,
        user_id: str = None,
    ) -> str:
        """List document libraries (drives) in a SharePoint site.

        Args:
            site_id: SharePoint site ID
            user_id: Microsoft email. Optional if already authenticated.
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.sharepoint_list_drives(uid, site_id)
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def sharepoint_list_files(
        site_id: str,
        drive_id: str = None,
        path: str = "/",
        top: int = 50,
        user_id: str = None,
    ) -> str:
        """List files in a SharePoint document library.

        Args:
            site_id: SharePoint site ID
            drive_id: Optional specific drive/library ID
            path: Folder path within the library
            top: Max items (default 50)
            user_id: Microsoft email. Optional if already authenticated.
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.sharepoint_list_files(uid, site_id, drive_id, path, top)
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def sharepoint_download_file(
        site_id: str,
        item_id: str,
        drive_id: str = None,
        user_id: str = None,
        session_id: str = "default",
    ) -> str:
        """Download a file from SharePoint and save it to the sandbox.

        Args:
            site_id: SharePoint site ID
            item_id: File item ID
            drive_id: Optional drive ID
            user_id: Microsoft email. Optional if already authenticated.
            session_id: Sandbox session ID (default 'default')
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.sharepoint_download(
                uid, site_id, item_id, drive_id,
                save_to_sandbox=True,
                session_id=session_id,
            )
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def sharepoint_upload_file(
        site_id: str,
        path: str,
        content_base64: str,
        drive_id: str = None,
        content_type: str = "application/octet-stream",
        user_id: str = None,
    ) -> str:
        """Upload a file to SharePoint (max 4MB simple upload).

        Args:
            site_id: SharePoint site ID
            path: Destination path (e.g. /General/report.xlsx)
            content_base64: File content as base64
            drive_id: Optional drive ID
            content_type: MIME type
            user_id: Microsoft email. Optional if already authenticated.
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.sharepoint_upload(
                uid, site_id, path, content_base64, drive_id, content_type
            )
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def sharepoint_search(
        site_id: str,
        query: str,
        drive_id: str = None,
        top: int = 25,
        user_id: str = None,
    ) -> str:
        """Search for files within a SharePoint site.

        Args:
            site_id: SharePoint site ID
            query: Search query
            drive_id: Optional drive ID to scope search
            top: Max results (default 25)
            user_id: Microsoft email. Optional if already authenticated.
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.sharepoint_search(uid, site_id, query, drive_id, top)
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def sharepoint_list_lists(
        site_id: str,
        user_id: str = None,
    ) -> str:
        """List SharePoint lists in a site.

        Args:
            site_id: SharePoint site ID
            user_id: Microsoft email. Optional if already authenticated.
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.sharepoint_list_lists(uid, site_id)
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    @mcp.tool()
    async def sharepoint_list_items(
        site_id: str,
        list_id: str,
        top: int = 50,
        user_id: str = None,
    ) -> str:
        """List items in a SharePoint list.

        Args:
            site_id: SharePoint site ID
            list_id: SharePoint list ID
            top: Max items (default 50)
            user_id: Microsoft email. Optional if already authenticated.
        """
        try:
            uid = await _resolve_user(user_id)
            result = await graph_client.sharepoint_list_items(uid, site_id, list_id, top)
            return json.dumps(result, indent=2)
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)}, indent=2)

    # v1.9.4: 22 tools (added resolve_share_link, made user_id optional everywhere)
    logger.info("Microsoft OneDrive + SharePoint tools registered (22 tools, v1.9.4)")
