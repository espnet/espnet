from __future__ import annotations


class DemoTestProvider:
    @staticmethod
    def build_model(config):
        _ = config

        def model(speech):
            return {"hyp": f"speech={int(speech is not None)}"}

        return model
