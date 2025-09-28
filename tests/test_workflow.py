import shutil
import subprocess
from pathlib import Path


def test_workflow():
    project_path = Path(__file__).parent / '_test_project'
    data_folder_path = project_path / 'data'
    if data_folder_path.exists():
        shutil.rmtree(data_folder_path)
    subprocess.run(['python', str(project_path / 'train.py')])
    shutil.rmtree(data_folder_path)


if __name__ == '__main__':
    test_workflow()
