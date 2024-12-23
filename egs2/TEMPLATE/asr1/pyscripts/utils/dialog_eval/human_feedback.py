import gradio as gr

disable_btn = gr.Button(interactive=False, visible=False)


def get_ip(request: gr.Request):
    if "cf-connecting-ip" in request.headers:
        ip = request.headers["cf-connecting-ip"]
    elif "x-forwarded-for" in request.headers:
        ip = request.headers["x-forwarded-for"]
        if "," in ip:
            ip = ip.split(",")[0]
    else:
        ip = request.client.host
    return ip


def natural_vote1_last_response(request: gr.Request):
    ip_address1 = get_ip(request)
    print(f"Very Natural (voted). ip: {ip_address1}")
    return (
        "Very Natural",
        ip_address1,
    ) + (disable_btn,) * 4


def natural_vote2_last_response(request: gr.Request):
    ip_address1 = get_ip(request)
    print(f"Somewhat Awkward (voted). ip: {ip_address1}")
    return (
        "Somewhat Awkward",
        ip_address1,
    ) + (disable_btn,) * 4


def natural_vote3_last_response(request: gr.Request):
    ip_address1 = get_ip(request)
    print(f"Very Awkward (voted). ip: {ip_address1}")
    return (
        "Very Awkward",
        ip_address1,
    ) + (disable_btn,) * 4


def natural_vote4_last_response(request: gr.Request):
    ip_address1 = get_ip(request)
    print(f"Unnatural (voted). ip: {ip_address1}")
    return (
        "Unnatural",
        ip_address1,
    ) + (disable_btn,) * 4


def relevant_vote1_last_response(request: gr.Request):
    ip_address1 = get_ip(request)
    print(f"Highly Relevant (voted). ip: {ip_address1}")
    return (
        "Highly Relevant",
        ip_address1,
    ) + (disable_btn,) * 4


def relevant_vote2_last_response(request: gr.Request):
    ip_address1 = get_ip(request)
    print(f"Partially Relevant (voted). ip: {ip_address1}")
    return (
        "Partially Relevant",
        ip_address1,
    ) + (disable_btn,) * 4


def relevant_vote3_last_response(request: gr.Request):
    ip_address1 = get_ip(request)
    print(f"Slightly Irrelevant (voted). ip: {ip_address1}")
    return (
        "Slightly Irrelevant",
        ip_address1,
    ) + (disable_btn,) * 4


def relevant_vote4_last_response(request: gr.Request):
    ip_address1 = get_ip(request)
    print(f"Completely Irrelevant (voted). ip: {ip_address1}")
    return (
        "Completely Irrelevant",
        ip_address1,
    ) + (disable_btn,) * 4
