from comfy_api.latest import io


class StringIntAddNood(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="noodles-StringIntAddNood",
            display_name="String Int Add",
            category="noodles/convert",
            inputs=[
                io.String.Input(
                    "in_a",
                    default="0",
                    multiline=False,
                    display_name="A",
                ),
                io.Int.Input(
                    "in_b",
                    default=0,
                    display_name="B",
                ),
            ],
            outputs=[
                io.Int.Output("result", display_name="Out"),
            ],
        )

    @classmethod
    def execute(cls, *, in_a: str, in_b: int):  # ty:ignore[invalid-method-override]
        try:
            int_a = int(in_a)
        except ValueError as e:
            raise ValueError(f"Could not convert '{in_a}' to an integer.") from e

        return io.NodeOutput(result=int_a + in_b)  # ty:ignore[unknown-argument]


__all__ = [
    StringIntAddNood,
]
