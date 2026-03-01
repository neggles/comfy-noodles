try:
    from rich.traceback import install as traceback_install

    _ = traceback_install(
        show_locals=True,
        suppress=["asyncio", "trio", "anyio", "aiohttp"],
    )
    del traceback_install
except ImportError:
    pass

from noodles import NoodlesExtension

extension = NoodlesExtension()

nodes = extension.get_node_list_sync()

for node in nodes:
    try:
        # try instantiating the node to check for any immediate errors
        instance = node()
        print(f"Successfully instantiated node: {node.__name__}")
    except Exception as e:
        print(f"Failed to instantiate node: {node.__name__}, Error: {e}")
