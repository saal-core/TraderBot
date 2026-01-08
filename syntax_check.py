
import sys
import compileall

files = [
    '/home/dev/Hussein/TraderBot/src/services/database_handler.py',
    '/home/dev/Hussein/TraderBot/api.py',
    '/home/dev/Hussein/TraderBot/src/services/task_executor.py'
]

for f in files:
    try:
        compileall.compile_file(f, force=True)
        print(f"Successfully compiled {f}")
    except Exception as e:
        print(f"Error compiling {f}: {e}")
        sys.exit(1)
