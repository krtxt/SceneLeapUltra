import subprocess


def get_git_head_hash():
    """Get current git commit hash"""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None 