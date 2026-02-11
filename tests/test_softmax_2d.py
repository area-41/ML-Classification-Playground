# Trecho de teste conceitual
output = model(input_data)
output.backward() # Gradiente do PyTorch

# Comparação
my_grad_w, my_grad_b = compute_gradients(input_data, torch.softmax(output, dim=1), labels)

print(f"Diferença nos pesos: {torch.abs(model[0].weight.grad.t() - my_grad_w).mean()}")
