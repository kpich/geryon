#./scripts/run.sh --workflow linear --model qwen2.5:14b

#./scripts/run.sh --model mistral-nemo

#./scripts/run.sh --model llama3.1:8b

#./scripts/run.sh --model llama3.1:8b --max-iterations 2


./scripts/run.sh \
    --provider aws_bedrock \
    --model us.anthropic.claude-opus-4-5-20251101-v1:0 \
    --aws-profile saml \
    --aws-region us-east-2 \
    --max-iterations 5 \
    --num-proposals 5 \
    --critic-cycles 1
#    -v

#    --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 \





# works
#    --model arn:aws:bedrock:us-east-2:490004633549:inference-profile/us.amazon.nova-premier-v1:0 \

# doesn't work
#    --model arn:aws:bedrock:us-east-2:490004633549:inference-profile/us.anthropic.claude-opus-4-6-v1 \
