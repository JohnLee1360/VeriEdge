from __future__ import annotations

import sys
from pathlib import Path


LIB_DIR = Path(__file__).resolve().parents[2] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from common import expand_prompt_id_spec, parse_prompts_from_markdown, select_prompts_by_ids, select_task_prompts  # noqa: E402
from crypto_utils import (  # noqa: E402
    decrypt_bytes_aes_gcm,
    decrypt_task_key_from_request,
    encrypt_bytes_aes_gcm,
    encrypt_task_key_for_provider,
    generate_rsa_keypair,
    generate_task_key,
)
from exo_state_utils import first_shard_provider, iter_model_instances, node_ip_map  # noqa: E402


def test_parse_and_select_prompts(tmp_path):
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("1. first\n2. second\n3. third\n", encoding="utf-8")
    prompts = parse_prompts_from_markdown(prompt_file)
    assert prompts == [(1, "first"), (2, "second"), (3, "third")]

    selected = select_task_prompts(prompts, question_count=2, seed=7)
    assert len(selected) == 2
    assert len({prompt_id for prompt_id, _ in selected}) == 2
    selected_by_ids = select_prompts_by_ids(prompts, prompt_ids=[3, 1])
    assert selected_by_ids == [(3, "third"), (1, "first")]
    assert expand_prompt_id_spec({"prompt_id_range": [2, 3]}) == [2, 3]


def test_crypto_roundtrip(tmp_path):
    private_key = tmp_path / "provider_private.pem"
    public_key = tmp_path / "provider_public.pem"
    generate_rsa_keypair(private_key, public_key)

    task_key = generate_task_key()
    encrypted_key = encrypt_task_key_for_provider(task_key, public_key)
    decrypted_key = decrypt_task_key_from_request(encrypted_key, private_key)
    assert decrypted_key == task_key

    plaintext = b'{"task_id":"task-001","prompts":[{"prompt_id":1,"content":"hello"}]}'
    encrypted = encrypt_bytes_aes_gcm(plaintext, task_key)
    restored = decrypt_bytes_aes_gcm(encrypted["ciphertext_b64"], encrypted["nonce_b64"], task_key)
    assert restored == plaintext


def test_first_shard_provider_resolution():
    state_obj = {
        "nodeNetwork": {
            "node-a": {"interfaces": [{"ipAddress": "192.168.31.52"}]},
            "node-b": {"interfaces": [{"ipAddress": "192.168.31.159"}]},
            "node-c": {"interfaces": [{"ipAddress": "192.168.31.83"}]},
        },
        "instances": {
            "inst-1": {
                "MlxRing": {
                    "shardAssignments": {
                        "modelId": "mlx-community/Qwen3-0.6B-8bit",
                        "nodeToRunner": {
                            "node-a": "runner-a",
                            "node-b": "runner-b",
                            "node-c": "runner-c",
                        },
                        "runnerToShard": {
                            "runner-a": {"MlxShard": {"deviceRank": 0, "startLayer": 0}},
                            "runner-b": {"MlxShard": {"deviceRank": 1, "startLayer": 12}},
                            "runner-c": {"MlxShard": {"deviceRank": 2, "startLayer": 24}},
                        },
                    }
                }
            }
        },
    }
    assert node_ip_map(state_obj)["node-a"] == ["192.168.31.52"]
    instances = iter_model_instances(state_obj, "mlx-community/Qwen3-0.6B-8bit")
    assert len(instances) == 1
    node_id, ip = first_shard_provider(instances[0][1], state_obj)
    assert node_id == "node-a"
    assert ip == "192.168.31.52"
