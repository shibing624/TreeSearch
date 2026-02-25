# 语音通话插件

语音通话插件提供了通过 Twilio 进行语音通话的能力，支持外呼和接听。

## 安装

### 方式一：通过 npm 安装（推荐）

```bash
npm install @example/voice-call-plugin
```

### 方式二：本地文件夹安装（开发用）

```bash
cd plugins/voice-call
npm link
```

## 配置

语音通话插件需要以下环境变量：

| 变量名 | 说明 | 必填 |
|--------|------|------|
| `TWILIO_ACCOUNT_SID` | Twilio 账户 SID | 是 |
| `TWILIO_AUTH_TOKEN` | Twilio 认证令牌 | 是 |
| `TWILIO_PHONE_NUMBER` | Twilio 电话号码 | 是 |
| `WEBHOOK_URL` | Webhook 回调地址 | 是 |

### Webhook 配置

可以使用 ngrok 或 Tailscale 暴露本地 webhook URL：

```bash
ngrok http 3000
```

将生成的公网 URL 填入 `WEBHOOK_URL` 环境变量。

## TTS 语音合成

支持以下 TTS 服务商：

- **Google TTS**：默认选项，支持多语言
- **Azure TTS**：高质量语音合成
- **ElevenLabs**：自然语音合成

配置示例：
```json
{
  "tts_provider": "google",
  "voice": "zh-CN-Standard-A",
  "language": "zh-CN"
}
```

## 外呼电话

通过 Agent 工具发起外呼：

```python
await agent.call("voice_call", {
    "action": "outbound",
    "to": "+8613800138000",
    "message": "您好，这是一条测试语音通话。"
})
```

## 接听电话

启用接听功能需要配置白名单：

```json
{
  "inbound_enabled": true,
  "allowlist": ["+8613800138000", "+8613900139000"]
}
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `voice call <number>` | 拨打电话 |
| `voice hangup` | 挂断当前通话 |
| `voice status` | 查看通话状态 |

## Gateway RPC 方法

| 方法 | 说明 |
|------|------|
| `voice.call` | 发起通话 |
| `voice.hangup` | 挂断通话 |
| `voice.transfer` | 转接通话 |
| `voice.status` | 查询状态 |
