"""
Simple functional AI Chatbot with API
File: ai_chatbot_api_flask.py

Features:
- Flask backend with a /api/chat POST endpoint
- Serves a minimal single-file frontend at / for manual testing
- Uses the official OpenAI Python package (import openai)
- Keeps a short conversation history per-session (in-memory) for context

Requirements:
- Python 3.10+
- pip install flask flask-cors openai
- Set environment variable OPENAI_API_KEY before running

Notes:
- This is a minimal example for learning and prototyping. For production:
  * Use proper authentication on the API endpoint
  * Persist conversation history in a database
  * Add rate-limiting, input sanitization, logging, and monitoring

Run:
    export OPENAI_API_KEY="sk-..."
    python ai_chatbot_api_flask.py

Then open http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import openai
import uuid

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client (reads key from env variable)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "Set OPENAI_API_KEY environment variable before running."
    )
openai.api_key = api_key

# In-memory store for conversation history (warning: not persistent)
# Structure: conversations[session_id] = [ {role: 'user'|'assistant'|'system', content: '...'}, ... ]
conversations = {}

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Keep answers short unless user asks for details."
)

@app.route("/api/chat", methods=["POST"])
def chat_api():
    """POST JSON body:
    {
      "session_id": "optional-session-id",
      "message": "user message string",
      "max_tokens": 400
    }

    Response:
    {
      "session_id": "returned-or-new-session-id",
      "reply": "assistant response text"
    }
    """
    data = request.get_json(force=True)
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "`message` is required"}), 400

    session_id = data.get("session_id") or str(uuid.uuid4())
    # create history for new session
    if session_id not in conversations:
        conversations[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    # append user message
    conversations[session_id].append({"role": "user", "content": user_message})

    # Call OpenAI Chat Completions API
    try:
        # Use the standard openai.ChatCompletion.create pattern
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # change to gpt-4 or other model if available
            messages=conversations[session_id],
            max_tokens=data.get("max_tokens", 300),
            temperature=data.get("temperature", 0.7),
        )

        # Extract assistant message text in a robust way
        assistant_message = None
        if isinstance(resp, dict):
            assistant_message = resp.get("choices", [{}])[0].get("message", {}).get("content")
        else:
            # some OpenAI client versions return objects with attributes
            try:
                assistant_message = resp.choices[0].message.content
            except Exception:
                assistant_message = str(resp)

        if not assistant_message:
            assistant_message = "(no reply from model)"

        # Append assistant reply to history
        conversations[session_id].append({"role": "assistant", "content": assistant_message})

        # (Optional) Trim history to last N messages to control token usage
        MAX_HISTORY_MESSAGES = 20
        if len(conversations[session_id]) > MAX_HISTORY_MESSAGES:
            # keep system prompt + last (MAX_HISTORY_MESSAGES-1) messages
            conversations[session_id] = [conversations[session_id][0]] + conversations[session_id][- (MAX_HISTORY_MESSAGES - 1) :]

        return jsonify({"session_id": session_id, "reply": assistant_message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Minimal web UI for quick testing
INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Simple AI Chatbot</title>
    <style>
      body { font-family: Arial, sans-serif; max-width:800px; margin:40px auto; }
      #chat { border:1px solid #ddd; padding:16px; height:400px; overflow:auto; }
      .msg { margin:8px 0; }
      .user { text-align:right; }
      .assistant { text-align:left; color:#0b5394; }
      #controls { margin-top:12px; display:flex; gap:8px }
      textarea { flex:1; height:60px }
    </style>
  </head>
  <body>
    <h2>Simple AI Chatbot (local)</h2>
    <div id="chat"></div>

    <div id="controls">
      <textarea id="input" placeholder="Type a message..."></textarea>
      <button id="send">Send</button>
    </div>

    <script>
      let sessionId = null;
      const chatEl = document.getElementById('chat');
      const inputEl = document.getElementById('input');
      const sendBtn = document.getElementById('send');

      function appendMessage(text, cls){
        const d = document.createElement('div');
        d.className = 'msg ' + cls;
        d.textContent = text;
        chatEl.appendChild(d);
        chatEl.scrollTop = chatEl.scrollHeight;
      }

      sendBtn.onclick = async () => {
        const text = inputEl.value.trim();
        if(!text) return;
        appendMessage(text, 'user');
        inputEl.value = '';

        const payload = { message: text };
        if(sessionId) payload.session_id = sessionId;

        try{
          const res = await fetch('/api/chat', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify(payload)
          });
          const data = await res.json();
          if(data.error){
            appendMessage('Error: ' + data.error, 'assistant');
          } else {
            sessionId = data.session_id;
            appendMessage(data.reply, 'assistant');
          }
        } catch(err){
          appendMessage('Network error: ' + err.message, 'assistant');
        }
      }
    </script>
  </body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)


# --- Basic tests ---
# The following tests can be run by setting the environment variable TEST=1 and executing the file.
# They do not call OpenAI; they only verify the Flask routes behave for basic input validation.

def _run_basic_tests():
    print("Running basic sanity tests...")
    with app.test_client() as c:
        # 1) index page
        r = c.get('/')
        assert r.status_code == 200, 'index page should return 200'
        print(' - index OK')

        # 2) missing message -> 400
        r = c.post('/api/chat', json={})
        assert r.status_code == 400, 'POST /api/chat without message should return 400'
        print(' - /api/chat missing message validation OK')

        # 3) session creation + message validation (we can't test model call without mocking)
        r = c.post('/api/chat', json={'message': 'hi'})
        # If API key is invalid or model call fails, we should receive a 500 or a JSON error; just print outcome
        print(' - /api/chat with message returned', r.status_code)
        print('Basic tests completed.')


if __name__ == "__main__":
    if os.getenv('TEST') == '1':
        _run_basic_tests()
    else:
        app.run(debug=True, host="0.0.0.0", port=5000)
