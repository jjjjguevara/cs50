#include <cs50.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

// prototypes
int lettercount(string text);
int wordcount(string text);
int sentencecount(string text);

// code
int main(void)
{
    // prompts user for input text
    string text = get_string("Please enter your text: ");

    // counts letters, words, sentences
    int letters = lettercount(text);
    int words = wordcount(text);
    int sentences = sentencecount(text);

    // Coleman-Liau index averages and formula
    float L = (float) letters / (float) words * 100;
    float S = (float) sentences / (float) words * 100;
    int index = round(0.0588 * L - 0.296 * S - 15.8);

    // prints rounded reading level
    if (index > 1 && index < 16)
    {
        printf("Reading level: Grade %i\n", index);
    }
    else if (index < 1)
    {
        printf("Reading level: Before Grade 1\n");
    }
    else if (index > 16)
    {
        printf("Reading level: Grade 16+\n");
    }
}

// functions
// calculates and returns letter count
int lettercount(string text)
{
    if (text == NULL)
    {
        return 0;
    }

    int n = 0;
    for (int i = 0; text[i] != '\0'; i++)
    {
        if (isalpha(text[i]))
        {
            n++;
        }
    }
    return n;
}

// calculates and returns word count
int wordcount(string text)
{
    if (text == NULL)
    {
        return 0;
    }

    int n = 0;
    for (int i = 0; text[i] != '\0'; i++)
    {
        if (isspace(text[i]))
        {
            n++;
        }
    }
    return n + 1;
}

// calculates and returns sentence count
int sentencecount(string text)
{
    if (text == NULL)
    {
        return 0;
    }

    int n = 0;
    for (int i = 0; text[i] != '\0'; i++)
    {
        if (text[i] == '.' || text[i] == '!' || text[i] == '?')
        {
            n++;
        }
    }
    return n;
}
