from pathlib import Path
from unittest import TestCase

from click.testing import CliRunner

from bundle_filtering import process_range_of_simulations_julia as prsj


class TestProcessRangeOfSimulations(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_dir = Path(__file__).resolve().parents[1]
        cls.grid_dir = Path(__file__).resolve().parent / 'grid_0_8'
        cls.ground_truth = cls.project_dir / 'data' / 'pl_detections.csv'

    @classmethod
    def tearDown(cls) -> None:
        for file in cls.grid_dir.iterdir():
            if file.is_file():
                file.unlink()

    def test_pickles(self):
        cli_runner = CliRunner()
        result = cli_runner.invoke(prsj.runner, ['--path', str(self.grid_dir), '--file-id', 'output',
                                                 '--groundtruth-path', str(self.ground_truth)])
        print(result)
        self.assertEqual(0, result.exit_code)


class TestCsvProcessor(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.processor = prsj.CSVProcessor()
        cls.project_dir = Path(__file__).resolve().parents[1]
        cls.grid_dir = Path(__file__).resolve().parent / 'grid_0_8'
        cls.ground_truth = cls.project_dir / 'data' / 'pl_detections.csv'

    def test_max_iter(self):
        iterations = self.processor._get_iterations(str((self.grid_dir / 'output')))
        self.assertEqual([101, 105, 136], iterations)

    def test_get_detected(self):
        path = str(self.grid_dir / 'output')
        iteration = 136
        detected = self.processor.get_detected(path, iteration)
        self.assertEqual(35, len(detected))
        self.assertEqual(217.3522, detected[-1])
