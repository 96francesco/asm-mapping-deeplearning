import time

def is_export_done(task, timeout=600):
      """
      Check if the GEE export task is done
      """
      elapsed = 0
      while elapsed < timeout:
            status = task.status()
            print(f"Task {status['id']} status: {status['state']}")
            if status['state'] == 'COMPLETED':
                  return True
            if status['state'] == 'FAILED':
                  print(f"Task {status['id']} failed: {status['error_message']}")
                  return False
            time.sleep(30)
            elapsed += 30
      print(f"Task {status['id']} timed out after {timeout} seconds")
      return False