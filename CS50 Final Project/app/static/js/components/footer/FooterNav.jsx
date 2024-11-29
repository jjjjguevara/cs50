import React from "react";

const FooterNav = () => {
  const menuBlocks = [
    {
      title: "Browse",
      items: [
        { text: "Table of Contents", href: "/contents" },
        { text: "What's New", href: "/new" },
        { text: "Random Entry", href: "/random" },
        { text: "Chronological", href: "/chronological" },
        { text: "Archives", href: "/archives" },
      ],
    },
    {
      title: "About",
      items: [
        { text: "Editorial Information", href: "/info" },
        { text: "About", href: "/about" },
        { text: "Editorial Board", href: "/board" },
        { text: "How to Cite", href: "/cite" },
        { text: "Contact", href: "/contact" },
      ],
    },
    {
      title: "Support",
      items: [
        { text: "Support Us", href: "/support" },
        { text: "Make a Donation", href: "/donate" },
        { text: "Custom PDFs", href: "/pdfs" },
      ],
    },
  ];

  return (
    <footer className="footer-content">
      <div className="footer-menu">
        {menuBlocks.map((block, index) => (
          <div key={index} className="menu-block">
            <h4>{block.title}</h4>
            <ul>
              {block.items.map((item, itemIndex) => (
                <li key={itemIndex}>
                  <a href={item.href}>{item.text}</a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="footer-bottom">
        <div className="copyright">
          © {new Date().getFullYear()} Documentation Project
        </div>
        <div className="attribution">Created with DITA-OT</div>
      </div>
    </footer>
  );
};

export default FooterNav;
