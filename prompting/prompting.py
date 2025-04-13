from openai import OpenAI
import pickle

# STEP 1: TOGETHER AI SETUP
client = OpenAI(
    api_key='8610a2090bd9613f87d8aeb83378a0d203e54864cf5c6071a27142f89a39366d',
    base_url="https://api.together.xyz/v1"
)

# STEP 2: MODEL CONFIG
model_name = "mistralai/Mistral-7B-Instruct-v0.2"


# STEP 3: DEFINE EPISODES (ADD ALL 10 HERE)
with open("episode_description_policy.pkl", "rb") as f:
    episodes = pickle.load(f)

# STEP 4: (OPTIONAL) RATIONALE GENERATOR - LEAVE BLANK FOR INFERENCE
def generate_rationale_for(desc):
    return ""  # Replace with rule-based or actual rationale if you want to pre-fill

# STEP 5: BUILD PROMPT FROM ONE EPISODE
# def build_prompt_from_episode(desc):
#     header = """You are an intelligent reasoning assistant that explains agent decisions in the OpenAI Taxi-v3 environment.

# In this environment:
# - The world is a 5x5 grid with four fixed landmarks:
#     R (Red): (0, 0), G (Green): (0, 4), Y (Yellow): (4, 0), B (Blue): (4, 3)
# - A taxi must pick up a passenger and drop them off at their destination.
# - The state includes the taxi’s position, the passenger’s location (or whether they’re in the taxi), and the destination.
# - The agent can take one of the following actions:
#     0 = South, 1 = North, 2 = East, 3 = West, 4 = Pickup, 5 = Dropoff

# Given a state and the action taken, explain the rationale behind the agent's choice and keep it brief.
# """

#     prompt = header

#     prompt += f"\n---\n{desc}\nRationale:"
    
#     prompt += "\n"  # Leave last rationale blank for model to complete

#     return prompt

# STEP 5 (UPDATED): BUILD FULL PROMPT FOR ONE EPISODE
def build_prompt_from_full_episode(episode_descs):
    header = """You are an intelligent reasoning assistant that explains agent decisions in the OpenAI Taxi-v3 environment.

In this environment:
- The world is a 5x5 grid with four fixed landmarks:
    R (Red): (0, 0), G (Green): (0, 4), Y (Yellow): (4, 0), B (Blue): (4, 3)
- A taxi must pick up a passenger and drop them off at their destination.
- The state includes the taxi’s position, the passenger’s location (or whether they’re in the taxi), and the destination.
- The agent can take one of the following actions:
    0 = South, 1 = North, 2 = East, 3 = West, 4 = Pickup, 5 = Dropoff

Given a state and the action taken, explain the rationale behind the agent's choice.
"""

    prompt = header + "\n"

    for i, desc in enumerate(episode_descs):
        prompt += f"\n{desc}\nRationale:\n"

    prompt += "\n"  # Final newline for model continuation
    return prompt

# # STEP 6: PICK AN EPISODE TO RUN
# for i in range(len(episodes)):
#     print(f"Episode {i+1}:\n")
#     chosen_episode = episodes[i]  # You can loop over episodes if needed
#     for i, action in enumerate(chosen_episode):
#         print(f'Action {i+1}:')
#         prompt_text = build_prompt_from_episode(action)

#         # STEP 7: SEND PROMPT TO TOGETHER AI
#         response = client.chat.completions.create(
#             model=model_name,
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant that explains reinforcement learning agent decisions."},
#                 {"role": "user", "content": prompt_text}
#             ],
#             temperature=0,
#             max_tokens=300
#         )

#         # STEP 8: PRINT MODEL'S RESPONSE
#         print("🤖 Model Rationale Output:\n")
#         print(response.choices[0].message.content)


# STEP 6 & 7 (UPDATED): RUN MODEL ON ENTIRE EPISODE AT ONCE
for i, episode in enumerate(episodes):
    # print(f"\n====================\nEpisode {i+1}\n====================")
    prompt_text = build_prompt_from_full_episode(episode)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that explains reinforcement learning agent decisions."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.7,
        max_tokens=1000  # Adjust if longer episodes
    )

    # STEP 8: PRINT THE FULL OUTPUT
    print("\n🤖 Model Rationale Output:\n")
    print(prompt_text)
    print(response.choices[0].message.content)
    break
