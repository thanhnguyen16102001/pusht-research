"""Show the original pretrained Push-T policy in a live simulator window.

Run from pusht-research in the robodiff environment:
    python -B src/run_pretrained_diffusion.py --continuous --render --realtime-plotting --pace
    python -B src/run_pretrained_diffusion.py --sampler ddim --inference-steps 20 --render --pace

The policy weights and normalizer come from the original low-dimensional
checkpoint. Sampling defaults to its full DDPM chain; opt-in DDIM supports
fewer denoising steps. The shared project runner provides visualization,
Q/Esc handling, continuous episodes and actual coverage metrics. It uses the
same legacy=False environment reset and >=95% target as the project's MPC,
rather than reproducing the original headless benchmark preset.
"""

try:
    from .pretrained_diffusion_policy import load_policy
    from .pusht_runtime import PROJECT_ROOT
    from .run_diffusion_policy import main as run_policy
except ImportError:
    from pretrained_diffusion_policy import load_policy
    from pusht_runtime import PROJECT_ROOT
    from run_diffusion_policy import main as run_policy


def main(argv=None):
    return run_policy(
        argv,
        policy_loader=load_policy,
        default_checkpoint=PROJECT_ROOT / "checkpoints/pretrained/PushT_original_lowdim.ckpt",
        default_output=PROJECT_ROOT / "results/pretrained_diffusion",
        default_target_coverage=0.95,
        sampler_choices=("ddpm", "ddim"),
        description=__doc__,
    )


if __name__ == "__main__":
    main()
