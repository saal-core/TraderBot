
import os
import sys
from dotenv import load_dotenv

# Add project root to path
# scripts/huggingface_check.py -> .. -> root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables
load_dotenv()

# Force provider to huggingface for this test
os.environ["LLM_PROVIDER"] = "huggingface"

# Import after setting env var
from src.config.llm_provider import get_llm, get_streaming_llm, print_provider_status, get_huggingface_config

def test_huggingface_integration():
    print("\n🧪 Testing HuggingFace Integration")
    print("=" * 50)
    
    # Check configuration
    print("Checking configuration...")
    print_provider_status()
    
    config = get_huggingface_config()
    if not config.get("api_key"):
        print("❌ Error: HUGGINGFACEHUB_API_TOKEN is not set in .env")
        print("Please set it and try again.")
        return

    # Test 1: Basic Invocation
    print("\n1. Testing Basic Invocation...")
    try:
        try:
            llm = get_llm(temperature=0.7)
            model_name = config.get('model_name')
            print(f"   Model: {model_name}")
            print("   Prompt: 'Hello, are you working?'")
            print("   Response: ", end="", flush=True)
            
            # Invoke
            response = llm.invoke("Hello, are you working? Keep it brief.")
            print(f"✅ {response.content}")
            
        except Exception as e:
            # Check for error message indicating provider failure or timeout which suggests model issue
            error_str = str(e).lower()
            if "error code:" in error_str or "status code: 4" in error_str or "unsupported" in error_str or "stopiteration" in error_str:
                print(f"\n⚠️  Error invoking configured model: {e}")
                print("   Falling back to 'meta-llama/Meta-Llama-3-8B-Instruct' for verification...")
                
                # Update global config for test
                os.environ["HUGGINGFACE_MODEL"] = "meta-llama/Meta-Llama-3-8B-Instruct"
                
                # Re-fetch LLM
                llm = get_llm(temperature=0.7)
                response = llm.invoke("Hello, are you working? Keep it brief.")
                print(f"✅ {response.content}")
            else:
                raise e

    except Exception as e:
        print(f"\n❌ Error during basic invocation: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Streaming
    print("\n2. Testing Streaming...")
    try:
        # Re-fetch LLM to pick up env var change
        model_name = os.getenv("HUGGINGFACE_MODEL", config.get('model_name'))
        stream_llm = get_streaming_llm(temperature=0.7)
        print(f"   Model: {model_name}")
        print("   Prompt: 'Count from 1 to 5.'")
        print("   Response: ", end="", flush=True)
        
        for chunk in stream_llm.stream("Count from 1 to 5."):
            print(chunk.content, end="", flush=True)
        print("\n   ✅ Streaming completed")
        
    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")

if __name__ == "__main__":
    test_huggingface_integration()
