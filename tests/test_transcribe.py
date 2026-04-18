from transcribe import pick_device


def test_pick_device_prefers_cuda(mocker):
    mocker.patch("transcribe.torch.cuda.is_available", return_value=True)
    mocker.patch("transcribe.torch.backends.mps.is_available", return_value=False)
    assert pick_device() == "cuda"


def test_pick_device_falls_back_to_mps(mocker):
    mocker.patch("transcribe.torch.cuda.is_available", return_value=False)
    mocker.patch("transcribe.torch.backends.mps.is_available", return_value=True)
    assert pick_device() == "mps"


def test_pick_device_falls_back_to_cpu(mocker):
    mocker.patch("transcribe.torch.cuda.is_available", return_value=False)
    mocker.patch("transcribe.torch.backends.mps.is_available", return_value=False)
    assert pick_device() == "cpu"
