import os
import json
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from passlib.context import CryptContext
from jose import JWTError, jwt

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-this-in-production-please")
ALGORITHM = "HS256"
EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Simple file-based user store (swap for a real DB in production)
USERS_FILE = Path(__file__).parent / "users.json"


def _load_users() -> dict:
    if USERS_FILE.exists():
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}


def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=EXPIRE_MINUTES))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


def register_user(email: str, password: str, name: str) -> dict:
    users = _load_users()
    email = email.lower().strip()
    if email in users:
        return {"success": False, "error": "Email already registered"}
    users[email] = {
        "email": email,
        "name": name,
        "hashed_password": hash_password(password),
        "created_at": datetime.utcnow().isoformat()
    }
    _save_users(users)
    return {"success": True, "email": email, "name": name}


def authenticate_user(email: str, password: str) -> Optional[dict]:
    users = _load_users()
    email = email.lower().strip()
    user = users.get(email)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return {"email": user["email"], "name": user["name"]}


def get_user(email: str) -> Optional[dict]:
    users = _load_users()
    user = users.get(email.lower().strip())
    if not user:
        return None
    return {"email": user["email"], "name": user["name"]}
