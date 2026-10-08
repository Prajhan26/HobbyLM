"use client";

import Image from "next/image";
import { useState } from "react";
import { founders } from "@/content/site";
import styles from "./page.module.css";

export function TeamProfileSelector() {
  const [active, setActive] = useState<(typeof founders)[number]["id"]>("prajhan");

  return (
    <div className={styles.stage}>
      <section className={styles.portraitField} aria-label="Fuel Labs co-founders">
        <span className={styles.signalOne} aria-hidden="true" />
        <span className={styles.signalTwo} aria-hidden="true" />
        <span className={styles.signalThree} aria-hidden="true" />
        <div className={styles.portraits}>
          {founders.map((founder) => (
            <figure
              className={`${styles.portrait} ${active === founder.id ? styles.portraitActive : styles.portraitInactive}`}
              key={founder.id}
            >
              <Image
                src={`/portraits/${founder.id}${active === founder.id ? "-active" : ""}.png`}
                width={256}
                height={254}
                alt={founder.alt}
                priority
              />
            </figure>
          ))}
        </div>
      </section>

      <section className={styles.selector} aria-label="Fuel Labs co-founders">
        <p>Co-founders</p>
        {founders.map((founder) => (
          <a
            className={styles.profileLink}
            data-active={active === founder.id}
            href={founder.xUrl}
            key={founder.id}
            target="_blank"
            rel="noreferrer"
            onMouseEnter={() => setActive(founder.id)}
            onFocus={() => setActive(founder.id)}
            onClick={() => setActive(founder.id)}
          >
            <span className={styles.founderName}>{founder.name}</span>
          </a>
        ))}
      </section>
    </div>
  );
}
