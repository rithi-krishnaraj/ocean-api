"""
Sign-in with GNPS2 accounts, and who may change data.

- POST /auth/login takes a GNPS2 username and password and checks them with GNPS2's own
  sign-in (ai.gnps2.org /login, then /me). The password is passed through once and never
  stored or logged. On success the API issues its own bearer token:
  `Authorization: Bearer <token>` on later requests.
- Being a GNPS2 user is not enough: the account must also be on this API's user list.
  Roles:
    admin     upload/replace/delete any dataset, and add, change or remove users
    uploader  upload datasets, and replace/delete the ones they uploaded
- Owners are admins named in the server config (OCEAN_ADMINS, comma-separated GNPS2
  usernames). They can't be removed through the API, so nobody can lock them out.

Reading data needs no sign-in.
"""
from __future__ import annotations

import hashlib
import http.cookiejar
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from importer import DATA_DIR

DB_PATH = DATA_DIR / "ocean.db"
GNPS2_AUTH_URL = os.environ.get("GNPS2_AUTH_URL", "https://ai.gnps2.org").rstrip("/")
OWNERS = {u.strip().lower() for u in os.environ.get("OCEAN_ADMINS", "").split(",") if u.strip()}
TOKEN_DAYS = float(os.environ.get("TOKEN_DAYS", "7"))
USERNAME_RE = re.compile(r"[A-Za-z0-9_.@+-]{1,128}")
Role = Literal["admin", "uploader"]

# ---------------------------------------------------------------------------
# Storage (SQLite in the data volume, so it survives rebuilds)
# ---------------------------------------------------------------------------

_init_lock = threading.Lock()
_initialized = False


def db() -> sqlite3.Connection:
    global _initialized
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    if not _initialized:
        with _init_lock:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    username   TEXT PRIMARY KEY,
                    role       TEXT NOT NULL CHECK (role IN ('admin', 'uploader')),
                    added_by   TEXT,
                    added_at   TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tokens (
                    token_hash TEXT PRIMARY KEY,
                    username   TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
            """)
            _initialized = True
    return con


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def role_of(username: str) -> Optional[str]:
    username = username.lower()
    if username in OWNERS:
        return "admin"
    with db() as con:
        row = con.execute("SELECT role FROM users WHERE username = ?", (username,)).fetchone()
    return row["role"] if row else None


# ---------------------------------------------------------------------------
# GNPS2 password check
# ---------------------------------------------------------------------------

class BadCredentials(Exception):
    pass


def _find_username(obj) -> Optional[str]:
    """The account name in GNPS2's /me reply (tolerates a nested {"user": {...}})."""
    if isinstance(obj, dict):
        for key in ("username", "user_name", "login", "user"):
            v = obj.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for v in obj.values():
            found = _find_username(v) if isinstance(v, dict) else None
            if found:
                return found
    return None


def gnps2_check_password(username: str, password: str) -> str:
    """Return the GNPS2 username if the password is right; raise BadCredentials if not,
    or HTTPException(502) if GNPS2 can't be reached."""
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    headers = {"content-type": "application/json", "User-Agent": "ocean-metabolome-api/0.3"}

    def call(method: str, path: str, body: Optional[dict] = None):
        req = urllib.request.Request(f"{GNPS2_AUTH_URL}{path}", method=method, headers=headers,
                                     data=json.dumps(body).encode() if body is not None else None)
        with opener.open(req, timeout=20) as resp:
            text = resp.read()
            return json.loads(text) if text else {}

    try:
        call("POST", "/login", {"username": username, "password": password})
        me = call("GET", "/me")
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 401, 403, 422):
            raise BadCredentials() from None
        raise HTTPException(502, f"GNPS2 sign-in failed ({exc.code}); try again later.") from None
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(502, f"Could not reach GNPS2 to check the password: {exc}") from None
    finally:
        try:  # end the GNPS2 session we just opened; we only needed it to check the password
            if jar:
                call("POST", "/logout", {})
        except Exception:
            pass
    return (_find_username(me) or username).strip()


# ---------------------------------------------------------------------------
# Tokens and the current user
# ---------------------------------------------------------------------------

@dataclass
class User:
    username: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


_bearer = HTTPBearer(auto_error=False, description="Token from POST /auth/login")


def current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> User:
    if creds is None:
        raise HTTPException(401, "Sign in first: POST /auth/login, then send 'Authorization: Bearer <token>'.",
                            headers={"WWW-Authenticate": "Bearer"})
    with db() as con:
        row = con.execute("SELECT username, expires_at FROM tokens WHERE token_hash = ?",
                          (_hash(creds.credentials),)).fetchone()
    if row is None or row["expires_at"] < time.time():
        raise HTTPException(401, "Token is invalid or expired; sign in again.", headers={"WWW-Authenticate": "Bearer"})
    role = role_of(row["username"])
    if role is None:
        raise HTTPException(403, f"{row['username']} no longer has access to this API.")
    return User(row["username"], role)


def require_uploader(user: User = Depends(current_user)) -> User:
    return user  # every listed user can upload


def require_admin(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(403, "Only admins can do this.")
    return user


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

router = APIRouter()


class LoginBody(BaseModel):
    username: str = Field(..., description="GNPS2 username")
    password: str = Field(..., description="GNPS2 password (checked by GNPS2, never stored)")


@router.post("/auth/login", tags=["auth"])
def login(body: LoginBody):
    """Sign in with a GNPS2 account. Returns a bearer token for the upload and admin
    endpoints. The account must be on this API's user list (see GET /users)."""
    if not USERNAME_RE.fullmatch(body.username.strip()) or not body.password:
        raise HTTPException(422, "Give a GNPS2 username and password.")
    try:
        username = gnps2_check_password(body.username.strip(), body.password).lower()
    except BadCredentials:
        time.sleep(1)  # slow down password guessing
        raise HTTPException(401, "GNPS2 username or password is incorrect.") from None
    role = role_of(username)
    if role is None:
        raise HTTPException(403, f"Signed in to GNPS2 as {username}, but {username} isn't allowed to "
                                 "upload to this API. Ask an admin to add you.")
    token = secrets.token_urlsafe(32)
    expires = time.time() + TOKEN_DAYS * 86400
    with db() as con:
        con.execute("DELETE FROM tokens WHERE expires_at < ?", (time.time(),))
        con.execute("INSERT INTO tokens VALUES (?, ?, ?, ?)", (_hash(token), username, time.time(), expires))
    return {"token": token, "token_type": "bearer", "username": username, "role": role,
            "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expires))}


@router.post("/auth/logout", tags=["auth"])
def logout(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer), user: User = Depends(current_user)):
    """Revoke this token."""
    with db() as con:
        con.execute("DELETE FROM tokens WHERE token_hash = ?", (_hash(creds.credentials),))
    return {"signed_out": user.username}


@router.get("/auth/me", tags=["auth"])
def me(user: User = Depends(current_user)):
    """Who this token belongs to, and their role."""
    return {"username": user.username, "role": user.role, "owner": user.username in OWNERS}


@router.get("/users", tags=["users"])
def list_users(_: User = Depends(require_admin)):
    """Everyone who may upload, with their role. Owners come from the server config."""
    with db() as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM users ORDER BY role, username")]
    owners = [{"username": u, "role": "admin", "owner": True, "added_by": "server config"}
              for u in sorted(OWNERS)]
    return owners + [{**r, "owner": False} for r in rows if r["username"] not in OWNERS]


class UserBody(BaseModel):
    role: Role = Field(..., description="admin: manage users and all datasets; uploader: upload datasets")


def _check_username(username: str) -> str:
    username = username.strip().lower()
    if not USERNAME_RE.fullmatch(username):
        raise HTTPException(422, "Not a valid GNPS2 username.")
    if username in OWNERS:
        raise HTTPException(409, f"{username} is an owner (set in the server config) and can't be changed here.")
    return username


def _admins_left(con: sqlite3.Connection, without: str) -> int:
    n = con.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND username != ?", (without,)).fetchone()[0]
    return n + len(OWNERS)


@router.put("/users/{username}", tags=["users"])
def put_user(username: str, body: UserBody, admin: User = Depends(require_admin)):
    """Add a GNPS2 user, or change their role. Takes effect immediately, including for
    tokens they already hold."""
    username = _check_username(username)
    with db() as con:
        old = con.execute("SELECT role FROM users WHERE username = ?", (username,)).fetchone()
        if old and old["role"] == "admin" and body.role != "admin" and _admins_left(con, username) == 0:
            raise HTTPException(409, "That would leave no admins.")
        if old:
            con.execute("UPDATE users SET role = ?, updated_at = ? WHERE username = ?", (body.role, _now(), username))
        else:
            con.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", (username, body.role, admin.username, _now(), _now()))
    return {"username": username, "role": body.role, "created": old is None}


@router.delete("/users/{username}", tags=["users"])
def delete_user(username: str, _: User = Depends(require_admin)):
    """Remove a user. Their tokens stop working; datasets they uploaded stay."""
    username = _check_username(username)
    with db() as con:
        old = con.execute("SELECT role FROM users WHERE username = ?", (username,)).fetchone()
        if not old:
            raise HTTPException(404, f"{username} is not on the user list.")
        if old["role"] == "admin" and _admins_left(con, username) == 0:
            raise HTTPException(409, "That would leave no admins.")
        con.execute("DELETE FROM users WHERE username = ?", (username,))
        con.execute("DELETE FROM tokens WHERE username = ?", (username,))
    return {"removed": username}
