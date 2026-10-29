"""Device context ownership checks without camera hardware."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import weakref

from gx4m_real.camera import orbbec


class DeviceContextTests(unittest.TestCase):
    def setUp(self):
        orbbec.sdk_context.cache_clear()

        class Devices:
            def __init__(self, context):
                self.context = weakref.ref(context)

            def get_count(self):
                if self.context() is None:
                    raise RuntimeError('SDK context was released')
                return 1

            def get_device_by_index(self, index):
                self.get_count()
                return SimpleNamespace(context=self.context, index=index)

            def get_device_by_serial_number(self, serial):
                self.get_count()
                return SimpleNamespace(context=self.context, serial=serial)

        class Context:
            def query_devices(self):
                return Devices(self)

        self.context_factory = Mock(side_effect=Context)
        fake_sdk = SimpleNamespace(Context=self.context_factory,
                                   OBLogLevel=SimpleNamespace(ERROR='error'))
        sdk_patch = patch.object(orbbec, 'sdk', return_value=fake_sdk)
        sdk_patch.start()
        self.addCleanup(sdk_patch.stop)
        self.addCleanup(orbbec.sdk_context.cache_clear)

    def test_context_stays_alive_and_is_shared(self):
        first = orbbec.open_device()
        second = orbbec.open_device()
        self.assertIsNotNone(first.context())
        self.assertIs(first.context(), second.context())
        self.assertEqual(self.context_factory.call_count, 1)

    def test_select_by_serial(self):
        device = orbbec.open_device(serial='camera-serial')
        self.assertEqual(device.serial, 'camera-serial')
        self.assertIsNotNone(device.context())

    def test_invalid_index(self):
        for index in (-1, 1):
            with self.subTest(index=index):
                with self.assertRaisesRegex(RuntimeError, 'not found'):
                    orbbec.open_device(index)


if __name__ == '__main__':
    unittest.main()
