"""
Test 3: Verify Sarvam API connection and LLM token streaming.
Run from e:\Bhasha-Stream-main with venv activated.
"""
import asyncio
import sys
import os

# Windows fix: websockets/aiohttp need SelectorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_llm():
    from app.services.llm import LLMService

    print("\n🔍 Testing Sarvam LLM connection...")
    print(f"   Endpoint : {__import__('app.core.config', fromlist=['settings']).settings.llm.endpoint_url}")
    print(f"   Model    : {__import__('app.core.config', fromlist=['settings']).settings.llm.model_name}")

    llm = LLMService()
    tokens = []

    print("\n📡 Streaming response for: 'Hello, say one sentence in Hinglish'")
    print("   Response: ", end="", flush=True)

    async for token in llm.generate("Hello, say one sentence in Hinglish"):
        print(token, end="", flush=True)
        tokens.append(token)

    full = "".join(tokens)
    print("\n")

    if not tokens:
        print("❌ FAIL — No tokens received. Check your API key and endpoint.")
        return False

    if "mock" in full.lower() or "नमस्ते" in full and len(full) < 20:
        print("⚠️  WARNING — Got mock response. LLM may not be connected.")
        return False

    print(f"✅ PASS — Received {len(tokens)} tokens from Sarvam API")
    await llm.close()
    return True


asyncio.run(test_llm())
