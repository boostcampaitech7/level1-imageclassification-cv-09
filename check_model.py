import torchvision.models as models

def simplified_model_structure(model):
    def get_children(module):
        return list(module.named_children())

    def print_structure(module, prefix=''):
        children = get_children(module)
        for name, child in children:
            if isinstance(child, models.convnext.CNBlock):
                print(f"{prefix}{name}: CNBlock")
            elif len(list(child.children())) > 0:
                print(f"{prefix}{name}:")
                print_structure(child, prefix + '  ')
            else:
                print(f"{prefix}{name}: {child.__class__.__name__}")

    print("ConvNeXt Base Structure:")
    print_structure(model)

# 모델 로드
model = models.convnext_base(pretrained=True)

# 간단한 구조 출력
simplified_model_structure(model)