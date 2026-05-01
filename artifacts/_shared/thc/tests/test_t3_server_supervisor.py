from __future__ import annotations

import unittest

from unittest.mock import patch

from t3_server_supervisor import T3ServerSupervisor


class T3ServerSupervisorTests(unittest.TestCase):
    def test_init_preserves_ports(self) -> None:
        supervisor = T3ServerSupervisor(
            local_python_bin="/tmp/python",
            mac_host="192.168.31.83",
            local_port=19001,
            linux_tunnel_port=19002,
            ping_timeout_seconds=33.0,
        )
        self.assertEqual(supervisor.mac_host, "192.168.31.83")
        self.assertEqual(supervisor.local_port, 19001)
        self.assertEqual(supervisor.linux_tunnel_port, 19002)
        self.assertEqual(supervisor.ping_timeout_seconds, 33.0)

    @patch("t3_server_supervisor._post_ping")
    def test_status_uses_configured_mac_host(self, post_ping) -> None:
        post_ping.return_value = {"ok": True}
        supervisor = T3ServerSupervisor(local_python_bin="/tmp/python", mac_host="192.168.31.83")
        supervisor.status()
        self.assertEqual(post_ping.call_args_list[1].args[0], "http://192.168.31.83:8311/ping")


if __name__ == "__main__":
    unittest.main()
