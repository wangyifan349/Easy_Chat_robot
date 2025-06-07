#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>

#define MAX_Q       7       /* 问答条目数（最后一项“其他问题”兜底） */
#define MAX_ANS     2       /* 每条条目最多答案数 */
#define MAX_LEN    256      /* 输入缓冲区最大长度（含终止符） */
#define DP_MAX     512      /* 动态规划二维表最大维度 */

/* ---------- 问答条目结构定义 ---------- */
typedef struct {
    const char *question;         /* 存储的问题模板 */
    const char *answers[MAX_ANS]; /* 对应答案列表，NULL 表示无更多答案 */
} QAEntry;

/* ---------- 问答“字典”初始化 ---------- */
static QAEntry qa_dict[MAX_Q] = {
    /* 0: 你叫什么名字？ */
    {
        "你叫什么名字？",
        { "我叫小问问。", "我是智能小问，随时为你服务。" }
    },
    /* 1: 打印 Hello World */
    {
        "打印 Hello World",
        {
            "#include <stdio.h>\n"
            "\n"
            "int main() {\n"
            "    printf(\"Hello World\\n\");\n"
            "    return 0;\n"
            "}\n",
            NULL
        }
    },
    /* 2: 示例循环 */
    {
        "示例循环",
        {
            "for (int i = 0; i < 5; ++i) {\n"
            "    printf(\"%d\\n\", i);\n"
            "}\n",
            NULL
        }
    },
    /* 3: 最大公约数 */
    {
        "最大公约数",
        {
            "int gcd(int a, int b) {\n"
            "    while (b) {\n"
            "        int t = b;\n"
            "        b = a % b;\n"
            "        a = t;\n"
            "    }\n"
            "    return a;\n"
            "}\n",
            NULL
        }
    },
    /* 4: 今天天气怎么样？ */
    {
        "今天天气怎么样？",
        {
          "抱歉，我无法获取实时天气，请去天气网站查询。",
          "我不知道外面的天气，但希望你今天心情晴朗！"
        }
    },
    /* 5: 最长公共子序列是什么？ */
    {
        "最长公共子序列是什么？",
        {
          "最长公共子序列是两个序列中同时为子序列的最长序列。",
          "LCS 是一类经典的字符串动态规划问题。"
        }
    },
    /* 6: 兜底条目：其他问题 */
    {
        "其他问题",
        { "抱歉，我不清楚如何回答。", NULL }
    }
};

/* ---------- 编辑距离（Levenshtein） ---------- */
/*
  计算字符串 a 与 b 的最小编辑距离：
   - 插入（cost=1）
   - 删除（cost=1）
   - 替换（相同 cost=0，否则 cost=1）
  时间、空间复杂度 O(n*m)。使用静态 dp 数组以避免频繁分配。
  若任意长度超 DP_MAX-1，返回一个极大值，确保不会被选中。
*/
int edit_distance(const char *a, const char *b) {
    static int dp[DP_MAX][DP_MAX];
    int n = (int)strlen(a);
    int m = (int)strlen(b);
    if (n >= DP_MAX || m >= DP_MAX) {
        /* 防止数组越界 */
        return 1000000;
    }

    /* 初始化 边界情况：空串到长度 i 的距离 */
    for (int i = 0; i <= n; ++i) dp[i][0] = i;
    for (int j = 0; j <= m; ++j) dp[0][j] = j;

    /* 主 DP 循环 */
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            int cost = (a[i-1] == b[j-1]) ? 0 : 1;
            int del = dp[i-1][j] + 1;
            int ins = dp[i][j-1] + 1;
            int sub = dp[i-1][j-1] + cost;
            /* 取三者最小 */
            dp[i][j] = del < ins ? (del < sub ? del : sub)
                                 : (ins < sub ? ins : sub);
        }
    }
    return dp[n][m];
}

/* ---------- 最长公共子序列长度（LCS） ---------- */
/*
  计算 a 和 b 的最长公共子序列长度。
  时间、空间复杂度 O(n*m)。同样使用静态 dp 数组防止频繁 malloc。
  若长度超过 DP_MAX-1，返回 0，确保此条目不被选中。
*/
int longest_common_subsequence(const char *a, const char *b) {
    static int dp[DP_MAX][DP_MAX];
    int n = (int)strlen(a);
    int m = (int)strlen(b);
    if (n >= DP_MAX || m >= DP_MAX) {
        /* 防止数组越界 */
        return 0;
    }

    /* 初始化 边界情况：任何与空串的 LCS 长度为 0 */
    for (int i = 0; i <= n; ++i) dp[i][0] = 0;
    for (int j = 0; j <= m; ++j) dp[0][j] = 0;

    /* 主 DP 循环 */
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (a[i-1] == b[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                /* 取上或左的最大值 */
                dp[i][j] = dp[i-1][j] > dp[i][j-1]
                           ? dp[i-1][j]
                           : dp[i][j-1];
            }
        }
    }
    return dp[n][j=0], dp[n][m];  /* 返回最终结果 */
}

/* ---------- 工具函数：trim 与 to_lower ---------- */
/*
  trim：去掉字符串 s 首尾的所有 isspace() 字符。
  in-place 操作，不会分配额外内存。安全调用 strlen 一次后使用索引。
*/
void trim(char *s) {
    int i = 0, j = (int)strlen(s) - 1;
    /* 找到左侧第一个非空白 */
    while (i <= j && isspace((unsigned char)s[i])) i++;
    /* 找到右侧最后一个非空白 */
    while (j >= i && isspace((unsigned char)s[j])) j--;
    /* 如果有需要，移动中间部分到头部 */
    if (i > 0 || j < (int)strlen(s) - 1) {
        int k = 0;
        for (int p = i; p <= j; ++p) {
            s[k++] = s[p];
        }
        s[k] = '\0';  /* 添加终止符 */
    }
}

/*
  to_lower：将 s 中每个字符转换为小写，使用 tolower 安全转换。
  in-place 操作，长度不会改变。
*/
void to_lower(char *s) {
    for (char *p = s; *p; ++p) {
        *p = (char)tolower((unsigned char)*p);
    }
}

/* ---------- 找最佳匹配条目 ---------- */
/*
  method = 0：用 edit_distance（距离越小匹配度越高）
  method = 1：用 LCS（长度越大匹配度越高）
  返回值：qa_dict 数组中最佳匹配的索引，若都不匹配则返回最后一个“其他问题”。
  算法安全性：遍历固定 MAX_Q 项，无越界风险。
*/
int best_match(const char *question, int method) {
    int best_idx = MAX_Q - 1;  /* 默认“其他问题”索引 */
    if (method == 0) {
        /* 编辑距离最小化 */
        int best_d = edit_distance(question, qa_dict[0].question);
        for (int i = 1; i < MAX_Q - 1; ++i) {
            int d = edit_distance(question, qa_dict[i].question);
            if (d < best_d) {
                best_d = d;
                best_idx = i;
            }
        }
    } else {
        /* LCS 最大化 */
        int best_l = longest_common_subsequence(question, qa_dict[0].question);
        for (int i = 1; i < MAX_Q - 1; ++i) {
            int l = longest_common_subsequence(question, qa_dict[i].question);
            if (l > best_l) {
                best_l = l;
                best_idx = i;
            }
        }
    }
    return best_idx;
}

/* ---------- 主函数 ---------- */
int main(void) {
    char buf[MAX_LEN];
    /* 设定随机种子：基于时间，避免每次启动回答序列固定 */
    srand((unsigned)time(NULL));

    printf("欢迎使用智能问答，输入“退出”或“exit”结束对话。\n");
    int method = 1;  /* 默认使用 LCS 方法（method=0 切换为编辑距离） */

    while (1) {
        printf("\n请输入你的问题：");
        if (!fgets(buf, sizeof(buf), stdin)) {
            /* 输入流结束或读取失败，安全退出 */
            break;
        }
        /* 去掉首尾空白并且处理空行 */
        trim(buf);
        if (buf[0] == '\0') {
            continue;  /* 忽略空输入 */
        }

        /* 检查是否为退出指令（忽略大小写） */
        char low[MAX_LEN];
        strncpy(low, buf, MAX_LEN - 1);
        low[MAX_LEN - 1] = '\0';  /* 确保以 NUL 终止 */
        to_lower(low);
        if (strcmp(low, "退出") == 0 ||
            strcmp(low, "exit") == 0 ||
            strcmp(low, "quit") == 0) {
            printf("再见！\n");
            break;
        }

        /* 找最佳匹配并随机选取一个答案 */
        int idx = best_match(buf, method);
        int cnt = 0;
        for (int i = 0; i < MAX_ANS; ++i) {
            if (qa_dict[idx].answers[i]) {
                cnt++;
            }
        }
        /* cnt 最多为 MAX_ANS，不会越界 */
        int pick = (cnt > 1) ? (rand() % cnt) : 0;

        /* 输出回答：答案保证非 NULL */
        printf("回答：\n%s\n", qa_dict[idx].answers[pick]);
    }

    return 0;
}






————————————————————————  
项目文件结构  
————————————————————————  
qa_bot.c         // 源代码文件  
Makefile         // 可选的自动化编译脚本  
README.txt       // 本说明文档  
————————————————————————  
1. 在 Linux 下用 GCC/Clang 编译  
————————————————————————  
1.1 打开终端，进入包含 qa_bot.c 的目录。  
1.2 运行以下命令：  
    gcc qa_bot.c -std=c11 -O2 -s -static -o qa_bot  
    说明：  
      -std=c11    使用 C11 标准  
      -O2         优化级别  
      -s          去掉符号表和调试信息  
      -static     静态链接，生成单文件可执行程序  
      -o qa_bot   指定输出文件名  
1.3 执行并测试：  
    ./qa_bot  
————————————————————————  
2. 在 Windows 下用 MinGW-w64（命令行）  
————————————————————————  
2.1 安装 MinGW-w64，并将其 bin 目录添加到系统 PATH。  
2.2 打开 CMD 或 PowerShell，切换到源文件目录。  
2.3 运行以下命令（64 位交叉编译示例）：  
    x86_64-w64-mingw32-gcc qa_bot.c -std=c11 -O2 -s -static -o qa_bot.exe  
2.4 双击或在命令行中执行：  
    qa_bot.exe  
————————————————————————  
3. 在 Windows 下用 Dev-C++（IDE）  
————————————————————————  
3.1 启动 Dev-C++，新建 Empty Project，语言选 C。  
3.2 在项目中添加 qa_bot.c 文件。  
3.3 打开 “Project Options” → 在 “Compiler” 里的 “Additional Command-line Options” 填入：  
      -std=c11 -O2 -s -static  
3.4 （可选）在 “Linker” 选项中也填上 -static。  
3.5 保存并编译，最终生成 qa_bot.exe。  
————————————————————————  
4. 可选：跨平台 Makefile  
————————————————————————  
将下面内容保存为 Makefile，与 qa_bot.c 放同一目录。然后在 Linux 或 MSYS2/MinGW Shell 中运行 `make` 即可
Makefile 内容（纯文本）：

    CC       := gcc
    SRC      := qa_bot.c
    TARGET   := qa_bot

    # Windows 平台判断
    ifeq ($(OS),Windows_NT)
      EXE     := .exe
      CC      := x86_64-w64-mingw32-gcc
    else
      EXE     :=
    endif

    CFLAGS   := -std=c11 -O2 -s -static
    LDFLAGS  :=

    .PHONY: all clean

    all: $(TARGET)$(EXE)

    $(TARGET)$(EXE): $(SRC)
        $(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

    clean:
        -rm -f $(TARGET)$(EXE)

使用方法：  
    make        # 生成 qa_bot（Linux）或 qa_bot.exe（Windows）  
    make clean  # 删除可执行文件  
————————————————————————  
5. 结果  
————————————————————————  
编译完成后，你将获得一个独立的、剥离了符号信息、静态链接的单文件可执行程序（qa_bot 或 qa_bot.exe）。  
无需携带额外的库或调试文件，直接分发即可
