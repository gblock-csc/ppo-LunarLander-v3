"""
月球着陆器 LunarLander‑v3完整项目
虚拟环境class1，python3.10
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from huggingface_sb3 import package_to_hub, load_from_hub


def test_random_env():
    print("=====阶段1：随机动作测试仿真环境====")
    env = gym.make("LunarLander-v3", render_mode="human")
    observation, info = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        print("Action taken:", action)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Environment is reset")
            observation, info = env.reset()
    env.close()


def train_ppo_agent():
    print("\n=====阶段2：PPO算法训练智能体====")
    env = make_vec_env("LunarLander-v3", n_envs=16)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1
    )
    model.learn(total_timesteps=int(2e5))
    model.save("ppo-LunarLander-v3")
    print("✅本地模型保存完成：ppo‑LunarLander‑v3.zip")
    return model


def upload_to_hub(model):
    print("\n=====阶段3：上传模型到HuggingFace Hub====")
    env_id = "LunarLander-v3"
    model_architecture = "PPO"
    # =========【重要！替换为你自己HuggingFace用户名】========
    repo_id = "qwe-321/ppo-LunarLander-v3"
    commit_message = "Upload PPO LunarLander‑v3 trained agent"
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
    package_to_hub(
        model=model,
        model_name="ppo-LunarLander-v3",
        model_architecture=model_architecture,
        env_id=env_id,
        eval_env=eval_env,
        repo_id=repo_id,
        commit_message=commit_message
    )
    print(f"✅模型推送到：https://huggingface.co/{repo_id}")


def evaluate_from_hub():
    print("\n=====阶段4：从Hub加载模型进行评估====")
    env_id = "LunarLander-v3"
    repo_id = "qwe-321/ppo-LunarLander-v3"
    filename = "ppo-LunarLander-v3.zip"
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    checkpoint = load_from_hub(repo_id, filename)
    model = PPO.load(checkpoint, custom_objects=custom_objects, print_system_info=True)
    env = gym.make(env_id, render_mode="human")
    observation, info = env.reset()
    for _ in range(1000):
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


if __name__ == "__main__":
    # 根据需求注释/取消注释
    test_random_env()
    trained_model = train_ppo_agent()
    upload_to_hub(trained_model)
    evaluate_from_hub()
