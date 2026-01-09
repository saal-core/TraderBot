
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables
load_dotenv()

# Force provider settings for local test
os.environ["LLM_PROVIDER"] = "huggingface"
os.environ["HUGGINGFACE_TYPE"] = "local"
# Use a small model for quick verification to avoid massive download/OOM
os.environ["HUGGINGFACE_MODEL"] = "antgroup/Agentar-Scale-SQL-Generation-32B"

from src.config.llm_provider import get_llm, print_provider_status

def test_local_huggingface():
    print("\n🧪 Testing Local HuggingFace Integration")
    print("=" * 50)
    
    # Check configuration
    print("Checking configuration...")
    print_provider_status()
    
    print(f"Target Model: {os.environ['HUGGINGFACE_MODEL']}")
    print("Note: This will download the model if not present. It uses ~2GB disk/RAM.")
    
    try:
        print("\n1. Loading Model Pipeline (this may take time)...")
        llm = get_llm(temperature=0.7)
        
        print("\n2. Generating Text...")
        print("   Prompt: 'Hello, how are you?'")
        print("   Response: ", end="", flush=True)
        
        # Invoke
        response = llm.invoke("Hello, how are you? Keep it brief.")
        print(f"✅\n{response}")
        
    except Exception as e:
        print(f"\n❌ Error during local execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_local_huggingface()
