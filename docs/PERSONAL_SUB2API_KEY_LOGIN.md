# Personal Sub2API key login

This optional integration lets a family member sign in to Open WebUI with an
API key issued by the same Sub2API service that serves their model requests.
It is intended for a small, private deployment, not for reselling access.

## Security model

- The browser submits the key to Open WebUI over HTTPS once, at sign-in.
- Open WebUI validates it server-to-server at Sub2API's identity endpoint.
- The key is stored only in Open WebUI's encrypted `oauth_session` storage.
  Its encryption uses `OAUTH_SESSION_TOKEN_ENCRYPTION_KEY`, or falls back to
  the stable `WEBUI_SECRET_KEY` when that variable is not configured.
- On each request to the exact configured Sub2API base URL, Open WebUI loads
  the current user's key. It never falls back to the administrator's shared
  service key for a mapped Sub2API user.
- The key is not added to browser local storage and is not written to logs.

## Required environment

Set these variables on the Open WebUI container. The base URL should be the
private Docker-network address used by Open WebUI, not a public DNS hostname.

```dotenv
ENABLE_SUB2API_KEY_LOGIN=true
SUB2API_KEY_LOGIN_BASE_URL=http://sub2api:8080
# Recommended: a distinct, stable Fernet-compatible secret for encrypted keys.
OAUTH_SESSION_TOKEN_ENCRYPTION_KEY=replace-with-a-long-random-secret
```

Configure the OpenAI-compatible connection in Open WebUI with the matching
base URL `http://sub2api:8080/v1`. The exact URL match prevents an individual
key from being accidentally sent to any other provider.

Keep `WEBUI_SECRET_KEY` stable across container recreations. Rotating the
encryption secret without first migrating stored sessions makes saved keys
unreadable and intentionally requires users to sign in again.

## First-time user flow

1. An administrator creates a Sub2API user and issues that user an API key.
2. The person selects **Sign in with a Sub2API key** in Open WebUI and pastes
   the key.
3. Open WebUI creates one regular local account for the stable Sub2API user
   ID. It uses a synthetic `@users.invalid` email and never copies the real
   Sub2API email address. It also writes the verified subject into protected
   account metadata; a pre-existing local account with the same predictable
   synthetic email is rejected instead of being linked.
4. The first external-key login never becomes an Open WebUI administrator.
   The normal Open WebUI default-user-role setting still applies; set it to
   `user` if family members should be usable immediately rather than pending
   admin approval.

## Rotation and revocation

- When a person signs in again with a replacement Sub2API key, Open WebUI
  replaces their encrypted credential.
- Disable, expire, or revoke a key in Sub2API to stop model calls. The next
  call is rejected by Sub2API.
- If the Open WebUI session is deleted, the user signs in with a valid key
  again. The server does not expose a key-recovery screen.

## Required Sub2API endpoint

The paired Sub2API change adds `GET /v1/sub2api/identity`, authenticated by
the existing API-key middleware. It returns only a stable numeric subject and
a display name, with `Cache-Control: no-store`. It does not return the API
key, email, balance, group, or billing data.

## 言川 AI 家庭入口

- 欢迎页会要求用户输入自己的 Sub2API Key，并可在首次登录时填写显示名称；该名称只保存在 Open WebUI，不会写回 Sub2API。
- 已使用 `Sub2API User <ID>` 占位名称创建的账号，下次填写显示名称后会自动替换；已有自定义名称不会被之后的 Key 登录覆盖，仍可在“账户 → 个人资料”中修改。
- 要让浏览器欢迎页只显示 Key 登录，设置持久化配置 `ui.enable_login_form = false`、`ui.enable_signup = false`，并保留 `ENABLE_SUB2API_KEY_LOGIN=true`。
- 建议保留 `ENABLE_PASSWORD_AUTH=true` 作为管理员受控恢复通道；若设为 `false`，所有本地密码登录（包括管理员）都会在服务端被拒绝。
- 默认允许同一 Key 在手机和电脑同时登录。若不同设备填入不同 Key，后一次登录会更新该用户保存的 Key，因此同一用户的设备应使用同一把有效 Key。

### 家庭模型列表

建议在 Sub2API 的家庭用户分组里配置 `/v1/models` 展示列表，保留：

1. `gpt-5.6-sol`：日常默认
2. `gpt-6-astra`：复杂任务
3. `gpt-5.6-luna`：快速轻量
4. `gpt-5.3-codex-spark`：编程

分组模型列表控制的是展示；它不是上游模型调用的硬权限控制。真正的模型授权和额度仍由 Sub2API 的账户、分组及网关规则负责。
