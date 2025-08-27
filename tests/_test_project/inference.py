from hurricore.utils import load_config


config = load_config('configs.debug')
ckpt_step: int = ...


def main():
    model_ckpt_path = config.PathConfig().checkpoints / f'ckpt_step_{ckpt_step}' / 'pytorch_model.bin'
    ...


if __name__ == '__main__':
    main()
