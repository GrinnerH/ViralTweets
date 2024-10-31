from transformers import AutoTokenizer, AutoConfig, AutoModel

import torch

class CustomModel(torch.nn.Module):
    """
    This takes a transformer backbone and puts a slightly-modified classification head on top.
    
    """

    def __init__(self, model_name, num_extra_dims, num_labels=2):
        # num_extra_dims corresponds to the number of extra dimensions of numerical/categorical data

        super().__init__()
        # 使用 AutoConfig 从指定的预训练模型加载配置，并设置标签数量。
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        # 加载指定的预训练 transformer 模型，使用前面获取的配置。
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        # 获取 transformer 的隐藏层大小，这取决于使用的模型。常见大小为 768 和 1024。查看 config.json 文件
        num_hidden_size = self.transformer.config.hidden_size # May be different depending on which model you use. Common sizes are 768 and 1024. Look in the config.json file 

        # 定义第一个全连接层 linear_layer_1，输入特征维度为 num_hidden_size + num_extra_dims，输出维度为 32。
        self.linear_layer_1 = torch.nn.Linear(num_hidden_size+num_extra_dims, 32)
        # Output size is 1 since this is a binary classification problem
        # 定义第二个全连接层 linear_layer_2，输入维度为 32，输出维度为 16。
        self.linear_layer_2 = torch.nn.Linear(32, 16)
        # 定义输出层 linear_layer_output，输入维度为 16，输出维度为 1（用于二分类的 logits 输出）。
        self.linear_layer_output = torch.nn.Linear(16, 1)
        # 定义激活函数为 Leaky ReLU，设置负半轴斜率为 0.6。
        self.relu = torch.nn.LeakyReLU(0.6)
        # 定义 dropout 层，随机丢弃 50% 的神经元，以防止过拟合。
        self.dropout_1 = torch.nn.Dropout(0.5)

    '''
    定义前向传播方法，接收以下参数：
    input_ids: 输入的 token ID。
    extra_features: 附加特征，形状为 [batch_size, dim]。
    attention_mask: 注意力掩码（可选），用于指定哪些 tokens 需要被注意。
    token_type_ids: token 类型 ID（可选），用于区分不同类型的输入。
    labels: 真实标签（可选），在训练过程中使用。
    '''
    # 定义前向传播方法，接收以下参数：
    def forward(self, input_ids, extra_features, attention_mask=None, token_type_ids=None, labels=None):
        """
        extra_features should be of shape [batch_size, dim] 
        where dim is the number of additional numerical/categorical dimensions(规模)
        """

        # 将输入数据传入 transformer 模型，获取隐藏状态，输出形状为 [batch_size, sequence_length, hidden_size]。
        hidden_states = self.transformer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # [batch size, sequence length, hidden size]

        # 获取每个输入序列的 CLS token 的嵌入，形状为 [batch_size, hidden_size]。
        cls_embeds = hidden_states.last_hidden_state[:, 0, :] # [batch size, hidden size]

        # 将 CLS token 的嵌入与附加特征拼接，形成新的输入特征，形状为 [batch_size, hidden_size + num_extra_dims]。
        concat = torch.cat((cls_embeds, extra_features), dim=-1) # [batch size, hidden size+num extra dims]

        # 将拼接后的特征通过第一个全连接层，应用激活函数，输出形状为 [batch_size, 32]。
        output_1 = self.relu(self.linear_layer_1(concat)) # [batch size, num labels]
        # 将第一个全连接层的输出通过第二个全连接层，应用激活函数，输出形状为 [batch_size, 16]。
        output_2 = self.relu(self.linear_layer_2(output_1))
        # 将第二个全连接层的输出通过输出层，获取最终输出，并应用 dropout，形状为 [batch_size, 1]
        final_output = self.dropout_1(self.linear_layer_output(output_2))

        return final_output