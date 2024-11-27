<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <xsl:output method="html" indent="yes"/>

    <!-- Root template -->
    <xsl:template match="/">
        <html>
            <head>
                <title><xsl:value-of select="//title"/></title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        max-width: 1000px;
                        margin: 40px auto;
                        padding: 20px;
                        color: #333;
                    }
                    .metadata-box {
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 20px;
                        margin: 20px 0;
                        background: #f8f9fa;
                    }
                    .metadata-table {
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }
                    .metadata-table th {
                        text-align: left;
                        padding: 12px;
                        background: #eef2f5;
                        border-bottom: 2px solid #ddd;
                        width: 200px;
                    }
                    .metadata-table td {
                        padding: 12px;
                        border-bottom: 1px solid #ddd;
                    }
                    .abstract {
                        margin: 30px 0;
                        padding: 20px;
                        background: #f8f9fa;
                        border-left: 4px solid #0066cc;
                    }
                    .keywords {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 8px;
                    }
                    .keyword {
                        background: #e9ecef;
                        padding: 4px 12px;
                        border-radius: 16px;
                        font-size: 0.9em;
                        color: #495057;
                    }
                    .references {
                        margin-top: 40px;
                        padding-top: 20px;
                        border-top: 2px solid #eee;
                    }
                    h1 {
                        color: #2c3e50;
                        border-bottom: 2px solid #eee;
                        padding-bottom: 10px;
                    }
                    h2 {
                        color: #34495e;
                        margin-top: 30px;
                    }
                    .citation-box {
                        background: #e9ecef;
                        padding: 15px;
                        border-radius: 4px;
                        margin: 20px 0;
                        font-family: monospace;
                    }
                </style>
            </head>
            <body>
                <main>
                    <xsl:apply-templates select="//title"/>

                    <!-- Metadata Section -->
                    <div class="metadata-box">
                        <table class="metadata-table">
                            <tr>
                                <th>Authors</th>
                                <td>
                                    <xsl:for-each select="//author">
                                        <xsl:value-of select="."/>
                                        <xsl:if test="position() != last()">, </xsl:if>
                                    </xsl:for-each>
                                </td>
                            </tr>
                            <tr>
                                <th>Institution</th>
                                <td><xsl:value-of select="//institution"/></td>
                            </tr>
                            <tr>
                                <th>Journal</th>
                                <td><xsl:value-of select="//othermeta[@name='journal']/@content"/></td>
                            </tr>
                            <tr>
                                <th>DOI</th>
                                <td><xsl:value-of select="//othermeta[@name='doi']/@content"/></td>
                            </tr>
                            <tr>
                                <th>Publication Date</th>
                                <td><xsl:value-of select="//othermeta[@name='publication-date']/@content"/></td>
                            </tr>
                            <tr>
                                <th>Categories</th>
                                <td>
                                    <xsl:for-each select="//category">
                                        <xsl:value-of select="."/>
                                        <xsl:if test="position() != last()">, </xsl:if>
                                    </xsl:for-each>
                                </td>
                            </tr>
                            <tr>
                                <th>Keywords</th>
                                <td>
                                    <div class="keywords">
                                        <xsl:for-each select="//keyword">
                                            <span class="keyword"><xsl:value-of select="."/></span>
                                        </xsl:for-each>
                                    </div>
                                </td>
                            </tr>
                        </table>

                        <div class="citation-box">
                            <strong>Citation: </strong>
                            <xsl:value-of select="//othermeta[@name='citation']/@content"/>
                        </div>
                    </div>

                    <!-- Abstract -->
                    <div class="abstract">
                        <h2>Abstract</h2>
                        <xsl:apply-templates select="//abstract"/>
                    </div>

                    <!-- Main Content -->
                    <xsl:apply-templates select="//body"/>

                    <!-- References -->
                    <div class="references">
                        <h2>References</h2>
                        <ol>
                            <xsl:for-each select="//reference">
                                <li><xsl:value-of select="."/></li>
                            </xsl:for-each>
                        </ol>
                    </div>
                </main>
            </body>
        </html>
    </xsl:template>

    <!-- Other templates -->
    <xsl:template match="section">
        <section>
            <h2><xsl:value-of select="title"/></h2>
            <xsl:apply-templates select="*[not(self::title)]"/>
        </section>
    </xsl:template>

    <xsl:template match="p">
        <p><xsl:apply-templates/></p>
    </xsl:template>

    <xsl:template match="ul">
        <ul><xsl:apply-templates/></ul>
    </xsl:template>

    <xsl:template match="li">
        <li><xsl:apply-templates/></li>
    </xsl:template>
</xsl:stylesheet>
