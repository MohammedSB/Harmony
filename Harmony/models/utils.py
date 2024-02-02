def get_embedding_size_from_arch(arch):
    if arch == "vit_tiny":
        return 192
    elif arch == "vit_small":
        return 384
    elif arch == "vit_base":
        return 768
    elif arch == "vit_large":
        return 1024