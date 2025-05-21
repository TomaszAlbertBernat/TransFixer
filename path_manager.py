import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PathManager:
    def __init__(self, audio_dir: str, transcriptions_dir: str, corrected_dir: str):
        """
        Manages the generation of standardized file and directory paths for the application.

        Args:
            audio_dir (str): Root directory for audio files.
            transcriptions_dir (str): Root directory for transcription files.
            corrected_dir (str): Root directory for corrected transcription files.
        """
        self.audio_dir = Path(audio_dir).resolve()
        self.transcriptions_dir = Path(transcriptions_dir).resolve()
        self.corrected_dir = Path(corrected_dir).resolve()

        logger.info(f"PathManager initialized with:")
        logger.info(f"  Audio Dir: {self.audio_dir}")
        logger.info(f"  Transcriptions Dir: {self.transcriptions_dir}")
        logger.info(f"  Corrected Dir: {self.corrected_dir}")
        
        # Ensure base directories exist upon initialization
        self.ensure_dir_exists(str(self.audio_dir))
        self.ensure_dir_exists(str(self.transcriptions_dir))
        self.ensure_dir_exists(str(self.corrected_dir))

    def ensure_dir_exists(self, dir_path: str) -> None:
        """Ensures the specified directory exists, creating it if necessary."""
        path = Path(dir_path)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {path}")
            except Exception as e:
                logger.error(f"Failed to create directory {path}: {e}", exc_info=True)
                # Depending on severity, could raise an error here
        elif not path.is_dir():
            logger.error(f"Path {path} exists but is not a directory.")
            # This is a critical error, should probably raise
            raise NotADirectoryError(f"Path {path} exists but is not a directory.")

    def get_relative_path(self, full_path: str, base_dir: Path) -> Path:
        """Calculates a relative path from a given base directory."""
        full_path_p = Path(full_path).resolve()
        base_dir_p = base_dir.resolve()
        try:
            relative_path = full_path_p.relative_to(base_dir_p)
            return relative_path
        except ValueError as e:
            logger.warning(f"Could not make {full_path_p} relative to {base_dir_p}: {e}. Using filename as fallback.")
            # Fallback if full_path is not inside base_dir (e.g. absolute path provided elsewhere)
            return Path(full_path_p.name)

    def get_transcription_path(self, audio_path: str) -> str:
        """
        Generates the corresponding transcription file path for a given audio file path.
        Maintains the relative subdirectory structure from the audio_dir.
        """
        audio_path_p = Path(audio_path).resolve()
        relative_to_audio_root = self.get_relative_path(str(audio_path_p), self.audio_dir)
        
        # Change extension to .txt
        transcription_filename = relative_to_audio_root.with_suffix(".txt")
        
        # Join with the base transcriptions directory
        full_trans_path = self.transcriptions_dir / transcription_filename
        self.ensure_dir_exists(str(full_trans_path.parent))
        return str(full_trans_path)

    def get_corrected_path(self, transcription_path: str) -> str:
        """
        Generates the corresponding corrected file path for a given transcription file path.
        Maintains the relative subdirectory structure from the transcriptions_dir.
        """
        trans_path_p = Path(transcription_path).resolve()
        relative_to_trans_root = self.get_relative_path(str(trans_path_p), self.transcriptions_dir)
        
        # Corrected files keep the same name (including .txt extension)
        # and are just placed in the corrected_dir with the same relative path.
        full_corrected_path = self.corrected_dir / relative_to_trans_root
        self.ensure_dir_exists(str(full_corrected_path.parent))
        return str(full_corrected_path)

    def get_lock_path(self, item_path: str, lock_type: str) -> str:
        """
        Generates a standardized lock file path for a given item and lock type.
        Lock files are created adjacent to the item they are locking.
        Example: /path/to/file.mp3 -> /path/to/file.mp3.transcription.lock
        """
        if not lock_type:
            raise ValueError("lock_type cannot be empty for get_lock_path")
        return f"{Path(item_path).resolve()}.{lock_type.lower()}.lock"
    
    # --- Optional: Methods to go "backwards" --- 
    def get_audio_path_from_transcription(self, transcription_path: str, 
                                          expected_audio_extensions: list[str] = None) -> str | None:
        """
        Attempts to find the original audio file path from a transcription file path.
        This is more complex due to potential variations in audio file extensions.

        Args:
            transcription_path (str): The path to the transcription file.
            expected_audio_extensions (list[str], optional): 
                A list of possible audio extensions (e.g., [".mp3", ".wav", ".m4a"]). 
                If None, common ones will be tried.

        Returns:
            str | None: The path to the corresponding audio file if found, else None.
        """
        if expected_audio_extensions is None:
            expected_audio_extensions = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]
        
        trans_path_p = Path(transcription_path).resolve()
        relative_to_trans_root = self.get_relative_path(str(trans_path_p), self.transcriptions_dir)

        # Base name without the .txt extension
        audio_basename_stem = relative_to_trans_root.stem
        parent_dir_in_audio = self.audio_dir / relative_to_trans_root.parent

        for ext in expected_audio_extensions:
            potential_audio_path = parent_dir_in_audio / (audio_basename_stem + ext)
            if potential_audio_path.exists() and potential_audio_path.is_file():
                return str(potential_audio_path)
        
        logger.warning(f"Could not find matching audio file for transcription {transcription_path} in {parent_dir_in_audio} with extensions {expected_audio_extensions}")
        return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    # Create a temporary directory for testing
    import tempfile
    import shutil
    test_dir_base_name = "path_manager_test_space"
    # Cleanup if exists from previous run
    if os.path.exists(test_dir_base_name):
        shutil.rmtree(test_dir_base_name)
    
    with tempfile.TemporaryDirectory(prefix=test_dir_base_name) as tmpdir_root:
        # Define test directories within the temp directory
        audio_root = Path(tmpdir_root) / "audio_files"
        trans_root = Path(tmpdir_root) / "text_transcriptions"
        corr_root  = Path(tmpdir_root) / "text_corrected"

        # PathManager will create these directories if they don't exist due to ensure_dir_exists in init
        # For explicit test, we can create them or let PathManager do it.
        # audio_root.mkdir(parents=True, exist_ok=True)
        # trans_root.mkdir(parents=True, exist_ok=True)
        # corr_root.mkdir(parents=True, exist_ok=True)

        pm = PathManager(str(audio_root), str(trans_root), str(corr_root))

        logger.info("--- Testing PathManager --- ")

        # Test 1: Basic audio to transcription
        test_audio1_abs = audio_root / "meeting_notes.mp3"
        test_audio1_abs.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists for dummy file
        test_audio1_abs.touch() # Create dummy audio file

        expected_trans1 = trans_root / "meeting_notes.txt"
        generated_trans1 = pm.get_transcription_path(str(test_audio1_abs))
        logger.info(f"Audio: {test_audio1_abs} -> Trans: {generated_trans1} (Expected: {expected_trans1})")
        assert Path(generated_trans1).resolve() == expected_trans1.resolve()
        assert expected_trans1.parent.exists() # Check dir creation

        # Test 2: Audio in subdirectory to transcription
        test_audio2_abs = audio_root / "project_alpha" / "user_interview.wav"
        test_audio2_abs.parent.mkdir(parents=True, exist_ok=True)
        test_audio2_abs.touch()
        
        expected_trans2 = trans_root / "project_alpha" / "user_interview.txt"
        generated_trans2 = pm.get_transcription_path(str(test_audio2_abs))
        logger.info(f"Audio: {test_audio2_abs} -> Trans: {generated_trans2} (Expected: {expected_trans2})")
        assert Path(generated_trans2).resolve() == expected_trans2.resolve()
        assert expected_trans2.parent.exists()

        # Test 3: Transcription to corrected
        # Note: generated_trans1 is already a string path from previous test
        expected_corr1 = corr_root / "meeting_notes.txt"
        generated_corr1 = pm.get_corrected_path(generated_trans1) # Use path from previous test
        logger.info(f"Trans: {generated_trans1} -> Corrected: {generated_corr1} (Expected: {expected_corr1})")
        assert Path(generated_corr1).resolve() == expected_corr1.resolve()
        assert expected_corr1.parent.exists()

        # Test 4: Transcription in subdirectory to corrected
        expected_corr2 = corr_root / "project_alpha" / "user_interview.txt"
        generated_corr2 = pm.get_corrected_path(generated_trans2)
        logger.info(f"Trans: {generated_trans2} -> Corrected: {generated_corr2} (Expected: {expected_corr2})")
        assert Path(generated_corr2).resolve() == expected_corr2.resolve()
        assert expected_corr2.parent.exists()

        # Test 5: Lock file paths
        item1_path = audio_root / "data.mp3"
        item1_path.touch()
        expected_lock1 = f"{item1_path.resolve()}.transcription.lock"
        generated_lock1 = pm.get_lock_path(str(item1_path), "transcription")
        logger.info(f"Item: {item1_path} (transcription) -> Lock: {generated_lock1} (Expected: {expected_lock1})")
        assert generated_lock1 == expected_lock1

        item2_path = trans_root / "project_beta" / "results.txt"
        item2_path.parent.mkdir(parents=True, exist_ok=True)
        item2_path.touch()
        expected_lock2 = f"{item2_path.resolve()}.correction.lock"
        generated_lock2 = pm.get_lock_path(str(item2_path), "correction")
        logger.info(f"Item: {item2_path} (correction) -> Lock: {generated_lock2} (Expected: {expected_lock2})")
        assert generated_lock2 == expected_lock2
        
        item3_path = corr_root / "final_report.txt"
        item3_path.touch()
        expected_lock3 = f"{item3_path.resolve()}.process.lock"
        generated_lock3 = pm.get_lock_path(str(item3_path), "process") # generic lock type
        logger.info(f"Item: {item3_path} (process) -> Lock: {generated_lock3} (Expected: {expected_lock3})")
        assert generated_lock3 == expected_lock3

        # Test 6: Audio from transcription (basic case)
        trans_for_audio_search = Path(generated_trans1) # meeting_notes.txt path
        audio_for_trans_search = pm.get_audio_path_from_transcription(str(trans_for_audio_search))
        logger.info(f"Trans: {trans_for_audio_search} -> Audio: {audio_for_trans_search} (Expected: {test_audio1_abs})")
        assert Path(audio_for_trans_search).resolve() == test_audio1_abs.resolve()
        
        # Test 7: Audio from transcription (subdirectory)
        trans_for_audio_search_sub = Path(generated_trans2) # project_alpha/user_interview.txt
        audio_for_trans_search_sub = pm.get_audio_path_from_transcription(str(trans_for_audio_search_sub))
        logger.info(f"Trans: {trans_for_audio_search_sub} -> Audio: {audio_for_trans_search_sub} (Expected: {test_audio2_abs})")
        assert Path(audio_for_trans_search_sub).resolve() == test_audio2_abs.resolve()

        # Test 8: Audio from transcription (file not found)
        non_existent_trans = trans_root / "no_such_audio_file.txt"
        non_existent_trans.parent.mkdir(parents=True, exist_ok=True)
        non_existent_trans.touch()
        audio_not_found = pm.get_audio_path_from_transcription(str(non_existent_trans))
        logger.info(f"Trans: {non_existent_trans} -> Audio: {audio_not_found} (Expected: None)")
        assert audio_not_found is None
        non_existent_trans.unlink()

        logger.info("PathManager tests completed successfully.")

    # Temporary directory and its contents are cleaned up automatically by TemporaryDirectory context manager
    # Manual cleanup for non-temporary test dirs if needed:
    # if os.path.exists(test_dir_base_name):
    #     shutil.rmtree(test_dir_base_name)
    #     logger.info(f"Cleaned up test directory: {test_dir_base_name}") 