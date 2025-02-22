from espnet2.sds.utils.chat import Chat


def test_append():
    chat = Chat(2)
    chat.init_chat({"role": "system", "content": "dummy one"})
    chat.append({"role": "user", "content": "dummy two"})
    chat_message = chat.to_list()
    assert len(chat_message) == 2
    assert chat_message[0] == {"role": "system", "content": "dummy one"}
    assert chat_message[1] == {"role": "user", "content": "dummy two"}
