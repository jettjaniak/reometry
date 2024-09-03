import torch
from reometry.hf_model import HFModel
from reometry import utils

def test_gpt2_patched_run():
    # Setup
    model_name = "gpt2"
    device = utils.get_device_str()
    hf_model = HFModel.from_model_name(model_name, device)
    
    # Test parameters
    seq_len = 5
    n_prompts = 6
    layer_pert = 0
    layer_read = -1
    batch_size = n_prompts
    # Generate input ids
    input_ids = utils.get_input_ids(
        chunk=0,
        seq_len=seq_len,
        n_prompts=n_prompts,
        tokenizer=hf_model.tokenizer,
    )

    # Run clean and patched versions
    clean_cache = hf_model.clean_run_with_cache(
        input_ids=input_ids, layers=[layer_pert, layer_read], batch_size=batch_size
    )
    
    pert_cache = hf_model.patched_run_with_cache(
        input_ids=input_ids,
        layer_write=layer_pert,
        pert_resid=clean_cache.resid_by_layer[layer_pert],
        layers_read=[layer_read],
        batch_size=batch_size,
    )

    # Assertions
    assert torch.allclose(
        clean_cache.resid_by_layer[layer_read],
        pert_cache.resid_by_layer[layer_read],
    ), "Residuals at the read layer should be identical"

    assert torch.allclose(
        clean_cache.logits, 
        pert_cache.logits, 
    ), "Logits should be identical"

    print("GPT-2 patched run test passed successfully!")