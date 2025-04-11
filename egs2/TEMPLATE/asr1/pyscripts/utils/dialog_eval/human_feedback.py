import gradio as gr

disable_btn = gr.Button(interactive=False, visible=False)


def get_ip(request: gr.Request) -> str:
    """
    Retrieve the IP address from an incoming HTTP request.

    Args:
        request (gr.Request):
            The incoming HTTP request from which the IP address will be extracted.

    Returns:
        str:
            The IP address as a string.
    """
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
    """
    Handle a user vote for naturalness as "Very Natural".


    Args:
        request (gr.Request):
            The Gradio request object providing access to HTTP headers and metadata.

    Returns:
        tuple:
            A tuple containing:
            ("Very Natural", <ip_address>, (disable_btn,) * 4)

            - "Very Natural": The selected vote or label.
            - <ip_address>: The IP address of the client retrieved from the request.
            - disable_btn: An object repeated four times,
            to disable natural vote buttons.
    """
    ip_address1 = get_ip(request)
    print(f"Very Natural (voted). ip: {ip_address1}")
    return (
        "Very Natural",
        ip_address1,
    ) + (disable_btn,) * 4


def natural_vote2_last_response(request: gr.Request):
    """
    Handle a user vote for naturalness as "Somewhat Awkward".


    Args:
        request (gr.Request):
            The Gradio request object providing access to HTTP headers and metadata.

    Returns:
        tuple:
            A tuple containing:
            ("Somewhat Awkward", <ip_address>, (disable_btn,) * 4)

            - "Somewhat Awkward": The selected vote or label.
            - <ip_address>: The IP address of the client retrieved from the request.
            - disable_btn: An object repeated four times,
            to disable natural vote buttons.
    """
    ip_address1 = get_ip(request)
    print(f"Somewhat Awkward (voted). ip: {ip_address1}")
    return (
        "Somewhat Awkward",
        ip_address1,
    ) + (disable_btn,) * 4


def natural_vote3_last_response(request: gr.Request):
    """
    Handle a user vote for naturalness as "Very Awkward".


    Args:
        request (gr.Request):
            The Gradio request object providing access to HTTP headers and metadata.

    Returns:
        tuple:
            A tuple containing:
            ("Very Awkward", <ip_address>, (disable_btn,) * 4)

            - "Very Awkward": The selected vote or label.
            - <ip_address>: The IP address of the client retrieved from the request.
            - disable_btn: An object repeated four times,
            to disable natural vote buttons.
    """
    ip_address1 = get_ip(request)
    print(f"Very Awkward (voted). ip: {ip_address1}")
    return (
        "Very Awkward",
        ip_address1,
    ) + (disable_btn,) * 4


def natural_vote4_last_response(request: gr.Request):
    """
    Handle a user vote for naturalness as "Unnatural".


    Args:
        request (gr.Request):
            The Gradio request object providing access to HTTP headers and metadata.

    Returns:
        tuple:
            A tuple containing:
            ("Unnatural", <ip_address>, (disable_btn,) * 4)

            - "Unnatural": The selected vote or label.
            - <ip_address>: The IP address of the client retrieved from the request.
            - disable_btn: An object repeated four times,
            to disable natural vote buttons.
    """
    ip_address1 = get_ip(request)
    print(f"Unnatural (voted). ip: {ip_address1}")
    return (
        "Unnatural",
        ip_address1,
    ) + (disable_btn,) * 4


def relevant_vote1_last_response(request: gr.Request):
    """
    Handle a user vote for relevance as "Highly Relevant".


    Args:
        request (gr.Request):
            The Gradio request object providing access to HTTP headers and metadata.

    Returns:
        tuple:
            A tuple containing:
            ("Highly Relevant", <ip_address>, (disable_btn,) * 4)

            - "Highly Relevant": The selected vote or label.
            - <ip_address>: The IP address of the client retrieved from the request.
            - disable_btn: An object repeated four times,
            to disable relevance vote buttons.
    """
    ip_address1 = get_ip(request)
    print(f"Highly Relevant (voted). ip: {ip_address1}")
    return (
        "Highly Relevant",
        ip_address1,
    ) + (disable_btn,) * 4


def relevant_vote2_last_response(request: gr.Request):
    """
    Handle a user vote for relevance as "Partially Relevant".


    Args:
        request (gr.Request):
            The Gradio request object providing access to HTTP headers and metadata.

    Returns:
        tuple:
            A tuple containing:
            ("Partially Relevant", <ip_address>, (disable_btn,) * 4)

            - "Partially Relevant": The selected vote or label.
            - <ip_address>: The IP address of the client retrieved from the request.
            - disable_btn: An object repeated four times,
            to disable relevance vote buttons.
    """
    ip_address1 = get_ip(request)
    print(f"Partially Relevant (voted). ip: {ip_address1}")
    return (
        "Partially Relevant",
        ip_address1,
    ) + (disable_btn,) * 4


def relevant_vote3_last_response(request: gr.Request):
    """
    Handle a user vote for relevance as "Slightly Irrelevant".


    Args:
        request (gr.Request):
            The Gradio request object providing access to HTTP headers and metadata.

    Returns:
        tuple:
            A tuple containing:
            ("Slightly Irrelevant", <ip_address>, (disable_btn,) * 4)

            - "Slightly Irrelevant": The selected vote or label.
            - <ip_address>: The IP address of the client retrieved from the request.
            - disable_btn: An object repeated four times,
            to disable relevance vote buttons.
    """
    ip_address1 = get_ip(request)
    print(f"Slightly Irrelevant (voted). ip: {ip_address1}")
    return (
        "Slightly Irrelevant",
        ip_address1,
    ) + (disable_btn,) * 4


def relevant_vote4_last_response(request: gr.Request):
    """
    Handle a user vote for relevance as "Completely Irrelevant".


    Args:
        request (gr.Request):
            The Gradio request object providing access to HTTP headers and metadata.

    Returns:
        tuple:
            A tuple containing:
            ("Completely Irrelevant", <ip_address>, (disable_btn,) * 4)

            - "Completely Irrelevant": The selected vote or label.
            - <ip_address>: The IP address of the client retrieved from the request.
            - disable_btn: An object repeated four times,
            to disable relevance vote buttons.
    """
    ip_address1 = get_ip(request)
    print(f"Completely Irrelevant (voted). ip: {ip_address1}")
    return (
        "Completely Irrelevant",
        ip_address1,
    ) + (disable_btn,) * 4
