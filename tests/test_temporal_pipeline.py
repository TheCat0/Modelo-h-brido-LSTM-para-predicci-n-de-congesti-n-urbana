import csv
import json
import tempfile
import unittest
from pathlib import Path

from hybrid_lstm_traffic import calculate_partition_sizes, save_artifacts


class TemporalSplitTests(unittest.TestCase):
    def test_partition_sizes_include_purged_gaps(self):
        train, validation, test, gap = calculate_partition_sizes(
            total_rows=500,
            time_steps=8,
            test_size=0.2,
            val_size=0.2,
        )

        self.assertEqual(gap, 7)
        self.assertEqual(train + validation + test + 2 * gap, 500)
        self.assertGreaterEqual(min(train, validation, test), 9)

        train_end = train
        validation_start = train_end + gap
        validation_end = validation_start + validation
        test_start = validation_end + gap
        self.assertEqual(validation_start - train_end, gap)
        self.assertEqual(test_start - validation_end, gap)

    def test_small_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Se requieren al menos"):
            calculate_partition_sizes(25, 8, 0.2, 0.2)

    def test_boyaca_partition_boundaries_do_not_overlap(self):
        dataset = Path(__file__).parents[1] / "data" / "traffic_observations_boyaca_guayaquil_2026.csv"
        with dataset.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        train, validation, test, gap = calculate_partition_sizes(len(rows), 8, 0.2, 0.2)
        train_indexes = set(range(0, train))
        validation_start = train + gap
        validation_indexes = set(range(validation_start, validation_start + validation))
        test_start = validation_start + validation + gap
        test_indexes = set(range(test_start, test_start + test))

        self.assertTrue(train_indexes.isdisjoint(validation_indexes))
        self.assertTrue(train_indexes.isdisjoint(test_indexes))
        self.assertTrue(validation_indexes.isdisjoint(test_indexes))
        self.assertEqual(min(validation_indexes) - max(train_indexes) - 1, gap)
        self.assertEqual(min(test_indexes) - max(validation_indexes) - 1, gap)


class FakeModel:
    def save(self, path):
        Path(path).write_text("model", encoding="utf-8")


class FakeJoblib:
    @staticmethod
    def dump(value, path):
        Path(path).write_text(str(value), encoding="utf-8")


class ArtifactPersistenceTests(unittest.TestCase):
    def test_all_inference_artifacts_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            save_artifacts(
                joblib=FakeJoblib,
                model=FakeModel(),
                sequence_scaler="sequence-scaler",
                context_preprocessor="context-preprocessor",
                metadata={"sequence_length": 8},
                artifacts_dir=directory,
            )

            output = Path(directory)
            self.assertTrue((output / "hybrid_lstm_traffic_model.keras").is_file())
            self.assertTrue((output / "sequence_scaler.joblib").is_file())
            self.assertTrue((output / "context_preprocessor.joblib").is_file())
            metadata = json.loads((output / "model_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["sequence_length"], 8)


if __name__ == "__main__":
    unittest.main()
