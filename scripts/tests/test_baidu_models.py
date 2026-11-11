import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import baidu_models


class DriveTokenTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'baidu-app.json').write_text(json.dumps({
            'app_key': 'test-client', 'secret_key': 'test-secret',
        }))
        self.token_path = self.root / 'baidu-token.json'
        self.token_path.write_text(json.dumps({'refresh_token': 'old-refresh'}))
        self.config = mock.patch.object(baidu_models, 'CONFIG', self.root)
        self.config.start()
        self.addCleanup(self.config.stop)

    @unittest.skipIf(os.name == 'nt', 'POSIX JSON token storage')
    def test_refresh_persists_rotated_token_with_private_permissions(self):
        response = mock.Mock()
        response.json.return_value = {
            'access_token': 'new-access', 'refresh_token': 'new-refresh',
        }
        with mock.patch.object(baidu_models.requests, 'post', return_value=response) as post:
            drive = baidu_models.Drive()
        self.assertEqual(drive.token['refresh_token'], 'new-refresh')
        self.assertEqual(json.loads(self.token_path.read_text()), drive.token)
        self.assertEqual(self.token_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(post.call_args.kwargs['data']['refresh_token'], 'old-refresh')
        self.assertEqual(list(self.root.glob('.baidu-token-*')), [])

    @unittest.skipIf(os.name == 'nt', 'POSIX JSON token storage')
    def test_failed_refresh_keeps_existing_token(self):
        original = self.token_path.read_bytes()
        response = mock.Mock()
        response.json.return_value = {'error': 'invalid_grant'}
        with mock.patch.object(baidu_models.requests, 'post', return_value=response):
            with self.assertRaisesRegex(RuntimeError, 'OAuth refresh failed'):
                baidu_models.Drive()
        self.assertEqual(self.token_path.read_bytes(), original)

    @unittest.skipIf(os.name == 'nt', 'POSIX JSON token storage')
    def test_missing_refresh_token_does_not_send_request(self):
        self.token_path.write_text('{}')
        with mock.patch.object(baidu_models.requests, 'post') as post:
            with self.assertRaisesRegex(RuntimeError, 'Missing refresh_token'):
                baidu_models.Drive()
        post.assert_not_called()

    def test_windows_keeps_dpapi_storage(self):
        path = self.root / 'baidu-token.dpapi'
        path.write_bytes(b'encrypted-token')
        response = mock.Mock()
        response.json.return_value = {'access_token': 'new-access', 'refresh_token': 'new-refresh'}
        with mock.patch.object(baidu_models, 'os', SimpleNamespace(name='nt')), \
                mock.patch.object(baidu_models, 'protect', side_effect=[b'{"refresh_token":"old-refresh"}', b'new-encrypted-token']) as protect, \
                mock.patch.object(baidu_models.requests, 'post', return_value=response):
            baidu_models.Drive()
        self.assertEqual(path.read_bytes(), b'new-encrypted-token')
        self.assertEqual(protect.call_args_list[0], mock.call(b'encrypted-token', True))


if __name__ == '__main__':
    unittest.main()
