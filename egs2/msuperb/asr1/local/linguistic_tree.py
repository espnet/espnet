import json
from typing import Dict, List, Union


class TreeNode(object):
    def __init__(self, name: str):
        self.name = name
        self.par = None
        self.level = 0

    def get_ancestors(self):
        if self.par is None:
            return [self]
        res = self.par.get_ancestors()
        res.append(self)
        return res


class InternalNode(TreeNode):
    def __init__(self, name: str):
        super().__init__(name)
        self.children = {}
        self.num = 0

    def add_child(self, node):
        self.children[node.name] = node
        if isinstance(node, InternalNode):
            self.num += node.num
        else:
            self.num += 1

    def remove_child(self, name: str):
        if name in self.children:
            if isinstance(self.children[name], InternalNode):
                self.num -= node.num
            else:
                self.num -= 1
        self.children.pop(name, None)

    def __str__(self) -> str:
        return (
            f"name: {self.name}\n"
            + f"par: {self.par.name if self.par is not None else None}\n"
            + f"level: {self.level}\n"
            + f"num: {self.num}\n"
            + f"children: {list(self.children.keys())}\n"
        )


class LeafNode(TreeNode):
    def __init__(self, name: str, iso: str = ""):
        super().__init__(name)
        self.iso = iso

    def __str__(self) -> str:
        return (
            f"name: {self.name}\n"
            + f"par: {self.par.name if self.par is not None else None}\n"
            + f"level: {self.level}\n"
            + f"iso: {self.iso}\n"
            + f"ancestors: {[node.name for node in self.get_ancestors()]}\n"
        )


class LanguageTree(object):
    nodes: List[Union[InternalNode, LeafNode]]
    iso2node: Dict[str, LeafNode]
    # name2node: Dict[str, Union[InternalNode, LeafNode]]

    def __init__(self):
        self.root = None
        self.iso2node = {}
        # self.name2node = {}  however name is not unique sometimes...

    def size(self):
        return len(self.iso2node)

    def build_from_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
        self.root = InternalNode(name="ROOT")
        self.__build(self.root, info)

    def __build(self, root: InternalNode, data):
        for k, v in data.items():
            if k == "lang":
                for d in v:
                    node = LeafNode(name=d["name"], iso=d["iso"])
                    node.par = root
                    node.level = root.level + 1
                    self.iso2node[d["iso"]] = node
                    # self.name2node[d["name"]] = node
                    root.add_child(node)
            else:
                node = InternalNode(name=k)
                node.par = root
                node.level = root.level + 1
                self.__build(node, v)
                # self.name2node[k] = node
                root.add_child(node)

    def get_node(self, tag: str, key: str):
        if tag == "iso":
            return self.iso2node[key]
        # elif tag == "name":
        #     return self.name2node[key]
        else:
            raise NotImplementedError(
                f'Searching not supported. Support tags: "iso", "name".'
            )

    def __str__(self) -> str:
        return (
            f"Total languages: {self.size()}\n"
            + f"Total families: {len(self.root.children)}\n"
        )

    def lca(self, iso1, iso2):
        node1 = self.iso2node[iso1]
        node2 = self.iso2node[iso2]
        if iso1 == iso2:
            return node1.level, node1.name

        anc1 = node1.get_ancestors()
        anc2 = node2.get_ancestors()
        st = 0
        while 1:  # Should correctly terminate if iso1 != iso2
            if anc1[st] == anc2[st]:
                st += 1
            else:
                break
        return st - 1, anc1[st - 1].name


if __name__ == "__main__":
    tree = LanguageTree()
    tree.build_from_json("linguistic.json")
    print(tree)
    node = tree.get_node("iso", "deu")
    print(node)
    print(tree.lca("eng", "deu"))
    with open("macro.json", "r", encoding="utf-8") as f:
        macro_info = json.load(f)
    # for k, v in macro_info.items():
    #     for iso in v:
    #         try:
    #             node = tree.get_node("iso", iso)
    #             print(node.get_ancestors()[1].name)
    #         except:
    #             print(f"Can not found {iso}.")
    #     input()
    print(tree.lca("eng", "pol"))
    print(tree.lca("ita", "pol"))
    print(tree.lca("dan", "kat"))
    print(tree.lca("jav", "bul"))
