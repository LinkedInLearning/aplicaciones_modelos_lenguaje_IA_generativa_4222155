import json
import os
import uuid
import logging
from datetime import datetime
from typing import Literal, Optional, Dict, Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("it_actions_mcp")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()  # stderr
logger.addHandler(handler)

mcp = FastMCP("ITActions")

DB_PATH = os.getenv("IT_REQUESTS_DB", "./it_requests.json")

Priority = Literal["low", "medium", "high"]
RequestType = Literal["software_install", "access_reset", "device_replacement"]

def _load_db() -> Dict[str, Any]:
    if not os.path.exists(DB_PATH):
        return {"requests": []}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_db(db: Dict[str, Any]) -> None:
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def _find_by_dedupe(db: Dict[str, Any], dedupe_key: str) -> Optional[Dict[str, Any]]:
    for r in db.get("requests", []):
        if r.get("dedupe_key") == dedupe_key:
            return r
    return None

@mcp.tool()
def create_it_request(
    request_type: RequestType,
    requester_email: str,
    summary: str,
    details: str,
    priority: Priority = "medium",
    dry_run: bool = True,
    dedupe_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Crea una solicitud IT (acción). Útil para integrarse con un ITSM real (ServiceNow/Jira SM).
    - dry_run=True: no escribe en DB, solo simula.
    - dedupe_key: evita duplicados (idempotencia por email_id por ejemplo).
    """
    db = _load_db()

    if dedupe_key:
        existing = _find_by_dedupe(db, dedupe_key)
        if existing:
            return {
                "status": "deduplicated",
                "request_id": existing["request_id"],
                "message": "Solicitud ya existía (dedupe_key).",
                "request": existing,
            }

    request = {
        "request_id": f"REQ-{uuid.uuid4().hex[:10]}",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "request_type": request_type,
        "requester_email": requester_email,
        "priority": priority,
        "summary": summary,
        "details": details,
        "state": "created",
        "dedupe_key": dedupe_key,
    }

    if dry_run:
        return {
            "status": "dry_run",
            "request_id": request["request_id"],
            "message": "Simulación: no se ha creado nada (dry_run=True).",
            "request": request,
        }

    db.setdefault("requests", []).append(request)
    _save_db(db)

    logger.info(f"[CREATE] {request['request_id']} type={request_type} priority={priority}")
    return {
        "status": "created",
        "request_id": request["request_id"],
        "message": "Solicitud creada correctamente.",
        "request": request,
    }

@mcp.tool()
def get_request_status(request_id: str) -> Dict[str, Any]:
    """Devuelve el estado de una solicitud existente."""
    db = _load_db()
    for r in db.get("requests", []):
        if r.get("request_id") == request_id:
            return {"found": True, "request": r}
    return {"found": False, "message": "No encontrada."}

if __name__ == "__main__":
    # Demo local: STDIO
    mcp.run(transport="stdio")
    