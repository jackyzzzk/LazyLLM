from typing import Dict, List, Any, Set
from pydantic import ConfigDict, Field, model_validator

from lazyllm.prompt_templates.base import BasePromptTemplate
from lazyllm.prompt_templates.prompt_template import PromptTemplate


class FewShotPromptTemplate(BasePromptTemplate):
    """少样本提示模板类，用于构建包含示例（few-shot examples）的结构化提示。

该模板由三部分组成：前缀（prefix）、多个格式化后的示例（examples）、后缀（suffix），
并通过指定的示例模板（egs_prompt_template）渲染每个示例。支持变量部分绑定（partial binding），
允许预先填充部分变量，剩余变量在最终调用 format 时提供。

模板变量必须被明确划分为 required_vars（运行时提供）和 partial_vars（预绑定），
二者之并集必须恰好等于 prefix 与 suffix 中出现的所有变量。

Attributes:
    prefix (str): 示例前的引导文本，可包含变量占位符
    suffix (str): 示例后的指令或问题文本，可包含变量占位符
    examples (List[Dict]): 示例列表，每个示例为字典，需匹配 egs_prompt_template 的变量
    egs_prompt_template (PromptTemplate): 用于格式化每个示例的子模板
    required_vars (List[str]): 最终 format 时必须提供的变量名列表
    partial_vars (Dict[str, Any]): 预绑定的变量字典，值可以是常量或无参可调用对象
    separator_for_egs (str): 示例之间的分隔符，默认为换行符 '\n'
"""
    model_config = ConfigDict(frozen=True)
    prefix: str = Field(..., description='The prefix text that comes before examples, it may include variables')
    suffix: str = Field(..., description='The suffix text that comes after examples, it may include variables')
    examples: List[Dict[str, Any]] = Field(default_factory=list, description='List of example dictionaries')
    egs_prompt_template: PromptTemplate = Field(..., description='Template for formatting each example')
    required_vars: List[str] = Field(
        default_factory=list, description='List of required variable names for the final prompt'
    )
    partial_vars: Dict[str, Any] = Field(
        default_factory=dict, description='Dictionary of partial variable functions or values'
    )
    separator_for_egs: str = Field(default='\n', description='The separator between examples')

    @model_validator(mode='after')
    def validate_variables(self):
        """模型验证器，在实例创建后自动调用，用于校验模板变量和示例的一致性。

执行以下检查：
1. partial_vars 中的所有键必须存在于 prefix 或 suffix 的模板变量中；
2. required_vars 与 partial_vars 不能有交集；
3. required_vars 和 partial_vars 的并集必须恰好等于 prefix 与 suffix 中出现的所有变量；
4. 每个示例字典必须包含 egs_prompt_template 所需的全部变量。

若任一检查失败，将抛出 ValueError。

此方法确保模板在使用前处于合法、自洽的状态。
"""
        # 1. All keys in partial_vars must exist in the combined prefix+suffix variables
        # 2. required_vars + partial_vars keys must exactly equal all template variables
        # 3. Examples must be compatible with egs_prompt_template

        # all variables: variables in prefix + suffix.
        # Note: egs_prompt_template variables are not included here.
        all_vars = self._get_all_variables()

        # Check if partial_vars.keys() exist in template variables
        partial_vars_keys = set(self.partial_vars.keys())
        invalid_partial_vars = partial_vars_keys - all_vars
        if invalid_partial_vars:
            raise ValueError(f'partial_vars contains variables not found in template: {invalid_partial_vars}')

        required_vars_set = set(self.required_vars)
        # Check if required_vars and partial_vars have overlap
        overlap_vars = required_vars_set & partial_vars_keys
        if overlap_vars:
            raise ValueError(f'required_vars and partial_vars have overlap: {overlap_vars}')

        # Check if required_vars + partial_vars keys exactly equal all template variables
        combined_vars = required_vars_set | partial_vars_keys
        if combined_vars != all_vars:
            missing_vars = all_vars - combined_vars
            extra_vars = combined_vars - all_vars
            error_msg = []
            if missing_vars:
                error_msg.append(f'Missing variables in required_vars or partial_vars: {missing_vars}')
            if extra_vars:
                error_msg.append(f'Extra variables not found in template: {extra_vars}')
            raise ValueError('; '.join(error_msg))

        # Check each example is compatible with egs_prompt_template
        if self.examples:
            egs_required_vars = set(self.egs_prompt_template.required_vars)
            for i, example in enumerate(self.examples):
                example_keys = set(example.keys())
                missing_egs_vars = egs_required_vars - example_keys
                if missing_egs_vars:
                    raise ValueError(f'Example {i} missing required variables : {missing_egs_vars}')
        return self

    def format(self, **kwargs) -> str:
        """根据提供的变量值生成完整的少样本提示字符串。

必须提供所有 required_vars 中声明的变量。partial_vars 中的变量会自动应用（若为可调用对象则执行），
并覆盖 kwargs 中同名的值。每个示例通过 egs_prompt_template 格式化后，按 separator_for_egs 拼接，
最终与 prefix 和 suffix 合并并填充变量。

Args:
    **kwargs: 包含所有 required_vars 的关键字参数

Returns:
    str: 完整渲染后的提示文本

Raises:
    KeyError: 缺少 required_vars 中的变量，或模板中存在未绑定的变量
    ValueError: 示例格式化失败或最终模板渲染出错
"""
        # Check if all required variables are provided
        missing_vars = set(self.required_vars) - set(kwargs.keys())
        if missing_vars:
            raise KeyError(f'Missing required variables: {missing_vars}')

        # 1. Apply partial variables. Note: Overrides the values in kwargs if var_name exists in partial_vars
        format_kwargs = {**kwargs}
        for var_name in self.partial_vars:
            try:
                # Apply partial variable function
                val = self.partial_vars[var_name]
                if callable(val):
                    format_kwargs[var_name] = val()
                else:
                    format_kwargs[var_name] = val
            except Exception as e:
                raise TypeError(f'Error applying partial function for variable "{var_name}": {e}')

        # 2. Format examples
        formatted_examples = []
        sep = self.separator_for_egs
        for example in self.examples:
            try:
                formatted_example = self.egs_prompt_template.format(**example)
                formatted_examples.append(formatted_example)
            except Exception as e:
                raise ValueError(f'Error formatting example {example}: {e}')

        # 3. Combine prefix, examples(already formatted), and suffix
        examples_text = sep.join(formatted_examples)

        # 4. Format the final prompt
        try:
            final_template = f'{self.prefix}\n{examples_text}\n{self.suffix}'
            return final_template.format(**format_kwargs)
        except KeyError as e:
            raise KeyError(f'Template variable not found: {e}')
        except Exception as e:
            raise ValueError(f'Error formatting template: {e}')

    def partial(self, **kwargs) -> 'FewShotPromptTemplate':
        """对模板进行部分变量绑定，返回一个新的 FewShotPromptTemplate 实例。

新实例中，传入的变量将从 required_vars 移至 partial_vars，后续调用 format 时无需再提供这些变量。
可用于逐步绑定变量或构建可复用的半成品模板。

Args:
    **kwargs: 要预绑定的变量名和值（值可为常量或无参函数）

Returns:
    FewShotPromptTemplate: 新的模板实例，已绑定指定变量

Raises:
    KeyError: 若 kwargs 中包含模板中不存在的变量名
"""
        # Check if all partial variables exist in the template
        all_vars = self._get_all_variables()

        invalid_vars = set(kwargs.keys()) - all_vars
        if invalid_vars:
            raise KeyError(f'Variables not found in template: {invalid_vars}')

        # Create new partial_vars dict with additional partial functions
        new_partial_vars = self.partial_vars.copy()
        new_required_vars = list(set(self.required_vars) - set(kwargs.keys()))
        for var_name, var_value in kwargs.items():
            new_partial_vars[var_name] = var_value

        # Create new template with updated partial_vars
        return FewShotPromptTemplate(
            prefix=self.prefix,
            suffix=self.suffix,
            examples=self.examples.copy(),
            egs_prompt_template=self.egs_prompt_template,
            required_vars=new_required_vars,
            partial_vars=new_partial_vars
        )

    def _get_all_variables(self) -> Set[str]:
        prefix_vars = set(BasePromptTemplate.get_template_variables(self.prefix))
        suffix_vars = set(BasePromptTemplate.get_template_variables(self.suffix))
        return prefix_vars | suffix_vars
