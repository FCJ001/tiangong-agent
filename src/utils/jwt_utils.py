from datetime import datetime, timedelta, timezone
import jwt

SECRET_KEY = "12af38e3ab85909849bfe0b89f89075d7677438a0f14c0304a46249ae513558d"
ALGORITHM = "HS256"

# 配置OAuth2 Bearer 模式
from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# 编码JWT，创建一个令牌。登录成功后，把用户信息传递过来。 生成的令牌以后发给前端。
# 前端每次请求都需要携带自己的授权令牌。
def encode_jwt(payload: dict) -> str:
    payload_copy = payload.copy()
    # 更新过期时间为30分钟后。注意使用 utc 时间
    payload_copy["exp"] = datetime.now(timezone.utc) + timedelta(minutes=30)
    payload_copy["iat"] = datetime.now(timezone.utc)

    print(f"准备encode payload {payload_copy}")

    token = jwt.encode(payload_copy, key =SECRET_KEY, algorithm=ALGORITHM)
    print(f"生成完成的token: {token}")
    return token


# 解码JWT，验证令牌的有效性。[从令牌中获取用户信息]
# 每次请求到达后端时，从请求头中提取令牌，然后调用此函数验证令牌的有效性。
# 返回 当时用户的信息（payload里面不要放敏感数据）
def verify_jwt(token: str) -> dict:
    try:
        payload = jwt.decode(token, key=SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise Exception("token已经过期")
    except jwt.InvalidTokenError:
        raise Exception("非法token")
    except Exception as e:
        raise Exception(f"token校验失败: {str(e)}")
