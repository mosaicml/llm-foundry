from llmfoundry.utils.builders import build_callback

def test_curriculum_learning_callback_builds():
    kwargs = {'dataset_index': 0}
    callback = build_callback('curriculum_learning', kwargs=kwargs, config={'train_loader': {}})
    assert callback is not None