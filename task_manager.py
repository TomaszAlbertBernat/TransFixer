import os
import logging
from pathlib import Path
from path_manager import PathManager

logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self, path_manager: PathManager, min_chars_for_valid_transcription: int = 50):
        """
        Manages task collection and lock files for transcription and correction.

        Args:
            path_manager (PathManager): Instance of PathManager for path operations.
            min_chars_for_valid_transcription (int): Minimum characters for a transcription to be considered valid.
        """
        self.path_manager = path_manager
        self.min_chars = min_chars_for_valid_transcription
        self._active_locks = set() # Internal tracking of locks created by this instance
        
        logger.info(f"TaskManager initialized with PathManager and min_chars={min_chars_for_valid_transcription}")

    def _get_lock_path(self, item_path: str, lock_type: str) -> str:
        """Helper to generate standardized lock file paths."""
        return self.path_manager.get_lock_path(item_path, lock_type)

    def acquire_lock(self, item_path: str, lock_type: str = "process") -> bool:
        """
        Acquires a lock for a given item.

        Args:
            item_path (str): The path to the item to lock (e.g., audio file or transcription file).
            lock_type (str): Type of lock (e.g., 'transcription', 'correction', 'process').
                             This helps in creating distinct lock files.

        Returns:
            bool: True if lock was acquired, False otherwise.
        """
        lock_file = self._get_lock_path(item_path, lock_type)
        if os.path.exists(lock_file):
            logger.warning(f"Lock file {lock_file} already exists for {item_path}. Cannot acquire lock.")
            return False
        try:
            Path(lock_file).touch()
            self._active_locks.add(lock_file)
            logger.debug(f"Acquired lock: {lock_file} for {item_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating lock file {lock_file} for {item_path}: {e}")
            return False

    def release_lock(self, item_path: str, lock_type: str = "process") -> None:
        """
        Releases a lock for a given item.

        Args:
            item_path (str): The path to the item whose lock is to be released.
            lock_type (str): Type of lock to release.
        """
        lock_file = self._get_lock_path(item_path, lock_type)
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
                logger.debug(f"Released lock: {lock_file} for {item_path}")
            else:
                logger.warning(f"Attempted to release non-existent lock: {lock_file} for {item_path}")
            if lock_file in self._active_locks:
                self._active_locks.remove(lock_file)
        except Exception as e:
            logger.error(f"Error removing lock file {lock_file} for {item_path}: {e}")

    def is_locked(self, item_path: str, lock_type: str = "process") -> bool:
        """
        Checks if an item is currently locked.

        Args:
            item_path (str): Path to the item.
            lock_type (str): Type of lock.

        Returns:
            bool: True if a lock file exists, False otherwise.
        """
        lock_file = self._get_lock_path(item_path, lock_type)
        return os.path.exists(lock_file)

    def cleanup_all_locks(self) -> None:
        """
        Cleans up all tracked lock files.
        Optionally, this could also scan directories for orphaned .lock files.
        """
        logger.info("Cleaning up all tracked lock files...")
        # Create a copy for iteration as items might be removed
        locks_to_remove = list(self._active_locks) 
        for lock_file in locks_to_remove:
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                    logger.info(f"Cleaned up lock file: {lock_file}")
                if lock_file in self._active_locks: # Check before removing from set
                    self._active_locks.remove(lock_file)
            except Exception as e:
                logger.error(f"Error cleaning up lock file {lock_file}: {e}")

    def _is_valid_transcription(self, file_path: str) -> bool:
        """Checks if a transcription file is valid (exists and meets min char count)."""
        if not os.path.exists(file_path):
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return len(content.strip()) >= self.min_chars
        except Exception as e:
            logger.error(f"Error reading transcription file {file_path} for validation: {e}")
            return False

    def collect_transcription_tasks(self) -> list[dict]:
        """
        Collects audio files needing transcription.

        Returns:
            list[dict]: A list of tasks, each a dict: {'audio_path': str, 'trans_path': str}
        """
        tasks = []
        logger.info(f"Scanning for audio files to transcribe in {self.path_manager.audio_dir}...")
        for root, _, files in os.walk(str(self.path_manager.audio_dir)):
            for file in files:
                if file.lower().endswith((".mp3", ".wav", ".m4a", ".flac")): # Common audio formats
                    audio_path = os.path.join(root, file)
                    trans_path = self.path_manager.get_transcription_path(audio_path)

                    if self.is_locked(audio_path, lock_type="transcription"):
                        logger.debug(f"Skipping task for {audio_path}: transcription lock exists.")
                        continue
                    
                    # Also check for a more generic process lock on the audio file itself
                    if self.is_locked(audio_path, lock_type="process"):
                         logger.debug(f"Skipping task for {audio_path}: generic process lock exists.")
                         continue

                    if not (os.path.exists(trans_path) and self._is_valid_transcription(trans_path)):
                        tasks.append({'audio_path': audio_path, 'trans_path': trans_path})
                        logger.debug(f"Found transcription task: {audio_path} -> {trans_path}")
                    else:
                        logger.debug(f"Skipping task for {audio_path}: valid transcription already exists at {trans_path}.")
        logger.info(f"Found {len(tasks)} potential transcription tasks.")
        return tasks

    def collect_correction_tasks(self) -> list[dict]:
        """
        Collects transcription files needing correction.

        Returns:
            list[dict]: A list of tasks, each a dict: {'trans_path': str, 'corrected_path': str}
        """
        tasks = []
        logger.info(f"Scanning for transcriptions to correct in {self.path_manager.transcriptions_dir}...")
        for root, _, files in os.walk(str(self.path_manager.transcriptions_dir)):
            for file in files:
                if file.lower().endswith(".txt"):
                    trans_path = os.path.join(root, file)
                    corrected_path = self.path_manager.get_corrected_path(trans_path)

                    if self.is_locked(trans_path, lock_type="correction"):
                        logger.debug(f"Skipping task for {trans_path}: correction lock exists.")
                        continue
                    
                    # Also check for a more generic process lock on the transcription file itself
                    if self.is_locked(trans_path, lock_type="process"):
                         logger.debug(f"Skipping task for {trans_path}: generic process lock exists.")
                         continue
                    
                    if not os.path.exists(corrected_path) and self._is_valid_transcription(trans_path):
                        # We only correct if the source transcription is valid
                        tasks.append({'trans_path': trans_path, 'corrected_path': corrected_path})
                        logger.debug(f"Found correction task: {trans_path} -> {corrected_path}")
                    elif not self._is_valid_transcription(trans_path):
                        logger.debug(f"Skipping correction for {trans_path}: source transcription is not valid.")
                    else: # Corrected path exists or other condition
                         logger.debug(f"Skipping correction for {trans_path}: corrected file already exists or source not valid.")
        logger.info(f"Found {len(tasks)} potential correction tasks.")
        return tasks

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    
    # Create dummy directories and files for testing
    test_base_dir = "task_manager_test_space"
    audio_dir_test = os.path.join(test_base_dir, "audio")
    trans_dir_test = os.path.join(test_base_dir, "transcriptions")
    corr_dir_test = os.path.join(test_base_dir, "corrected")

    # Cleanup previous test run
    import shutil
    if os.path.exists(test_base_dir):
        shutil.rmtree(test_base_dir)

    os.makedirs(os.path.join(audio_dir_test, "subdir"), exist_ok=True)
    os.makedirs(os.path.join(trans_dir_test, "subdir"), exist_ok=True)
    os.makedirs(os.path.join(corr_dir_test, "subdir"), exist_ok=True)

    # Create some dummy files
    Path(os.path.join(audio_dir_test, "audio1.mp3")).touch()
    Path(os.path.join(audio_dir_test, "subdir", "audio2.wav")).touch()
    Path(os.path.join(audio_dir_test, "audio3_locked.m4a")).touch()
    # Create a lock for audio3
    Path(os.path.join(audio_dir_test, "audio3_locked.m4a.transcription.lock")).touch()

    # Create a dummy valid transcription that doesn't need correction yet
    valid_trans_path = os.path.join(trans_dir_test, "audio1.txt")
    with open(valid_trans_path, "w") as f:
        f.write("This is a perfectly valid transcription of sufficient length.")

    # Create a short transcription that needs correction (but is too short by default min_chars)
    short_trans_path = os.path.join(trans_dir_test, "subdir", "audio2.txt")
    with open(short_trans_path, "w") as f:
        f.write("Too short.")
        
    # Create a valid transcription that IS corrected
    Path(os.path.join(trans_dir_test, "audio4_already_corrected.txt")).write_text("This is valid and needs correcting.")
    Path(os.path.join(corr_dir_test, "audio4_already_corrected.txt")).write_text("This is the correction.")

    # Initialize PathManager and TaskManager
    path_manager = PathManager(audio_dir_test, trans_dir_test, corr_dir_test)
    tm = TaskManager(path_manager, min_chars_for_valid_transcription=20)

    logger.info("--- Testing Transcription Task Collection ---")
    trans_tasks = tm.collect_transcription_tasks()
    logger.info(f"Collected transcription tasks: {trans_tasks}")

    logger.info("--- Testing Correction Task Collection ---")
    corr_tasks = tm.collect_correction_tasks()
    logger.info(f"Collected correction tasks: {corr_tasks}")

    logger.info("--- Testing Lock Management ---")
    test_item = os.path.join(audio_dir_test, "test_lock_item.flac")
    Path(test_item).touch()

    logger.info(f"Is {test_item} locked (process)? {tm.is_locked(test_item, 'process')}")
    assert not tm.is_locked(test_item, 'process')
    
    logger.info(f"Acquiring process lock for {test_item}")
    assert tm.acquire_lock(test_item, 'process')
    logger.info(f"Is {test_item} locked (process)? {tm.is_locked(test_item, 'process')}")
    assert tm.is_locked(test_item, 'process')
    
    logger.info(f"Attempting to re-acquire process lock for {test_item} (should fail)")
    assert not tm.acquire_lock(test_item, 'process')
    
    logger.info(f"Releasing process lock for {test_item}")
    tm.release_lock(test_item, 'process')
    logger.info(f"Is {test_item} locked (process)? {tm.is_locked(test_item, 'process')}")
    assert not tm.is_locked(test_item, 'process')

    # Test cleanup
    tm.cleanup_all_locks()
    logger.info(f"Active locks after cleanup: {tm._active_locks}")
    assert not tm._active_locks
    # Check if physical lock files are gone
    assert not os.path.exists(os.path.join(audio_dir_test, "audio3_locked.m4a.transcription.lock"))
    assert not os.path.exists(test_item + ".process.lock")
    
    logger.info("TaskManager example usage complete.")
    logger.info(f"Test space at {test_base_dir} can be manually inspected or deleted.") 