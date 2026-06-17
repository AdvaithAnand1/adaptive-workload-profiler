import unittest

from monitor import FEATURE_NAMES, _parse_typeperf_gpu_output


class MonitorTests(unittest.TestCase):
    def test_gpu_feature_schema_is_fixed(self):
        self.assertEqual(len(FEATURE_NAMES), 29)
        self.assertEqual(FEATURE_NAMES[-1], "GPU Available [0/1]")

    def test_parses_windows_gpu_counters(self):
        output = '''"(PDH-CSV 4.0)","\\\\PC\\GPU Engine(x_engtype_3D)\\Utilization Percentage","\\\\PC\\GPU Engine(x_engtype_Video Decode)\\Utilization Percentage","\\\\PC\\GPU Adapter Memory(x)\\Dedicated Usage","\\\\PC\\GPU Adapter Memory(x)\\Shared Usage"
"time","12.5","34.5","104857600","209715200"
'''
        metrics = _parse_typeperf_gpu_output(output)
        self.assertEqual(metrics.usage_pct, 34.5)
        self.assertEqual(metrics.usage_3d_pct, 12.5)
        self.assertEqual(metrics.usage_video_pct, 34.5)
        self.assertEqual(metrics.dedicated_memory_mb, 100.0)
        self.assertEqual(metrics.shared_memory_mb, 200.0)
        self.assertEqual(metrics.available, 1.0)


if __name__ == "__main__":
    unittest.main()
